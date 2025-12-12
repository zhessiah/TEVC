import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.enriched_qmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from utils.attack_util import logits_margin, attack, get_diff, noise_atk, get_max_diff
# from torch.autograd import Variable
# from utils.pareto import MinNormSolver, gradient_normalizers
from controllers import REGISTRY as mac_REGISTRY


class MARCOLearner:
    def __init__(self, genome, pop_genome, scheme, logger, args):
        self.args = args
        self.Genome = genome  # Single Genome (双MAC) for RL training
        self.logger = logger
        self.scheme = scheme  # Save scheme for creating population
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(genome.parameters())  # This includes both mac1 and mac2 parameters

        if args.mixer == "qatten":
            self.mixer = QattenMixer(args)
        elif args.mixer == "vdn":
            self.mixer = VDNMixer()
        elif args.mixer == "qmix":
            self.mixer = Mixer(args)
        else:
            raise "mixer error"
        self.target_mixer = copy.deepcopy(self.mixer)
        self.params += list(self.mixer.parameters())

        print('Mixer Size: ')
        print(get_parameters_num(self.mixer.parameters()))
        
        if self.args.weight_td:
            self.log_var_a = th.zeros((1,), requires_grad=True, device=self.device)
            self.log_var_b = th.zeros((1,), requires_grad=True, device=self.device)
            self.log_var_b.data.fill_(self.args.robust_lambda)
            self.params = ([p for p in self.params] + [self.log_var_a] + [self.log_var_b])
            
        elif self.args.weight_adv_loss:
            self.log_var_b = th.zeros((1,), requires_grad=True, device=self.device)
            self.log_var_b.data.fill_(self.args.robust_lambda) 
            self.numerator = self.args.robust_lambda // 10
            self.params = ([p for p in self.params]+ [self.log_var_b])
        

        if self.args.optimizer == 'adam':
            self.optimiser = Adam(params=self.params,  lr=args.lr, weight_decay=getattr(args, "weight_decay", 0))
        else:
            self.optimiser = RMSprop(params=self.params, lr=args.lr, alpha=args.optim_alpha, eps=args.optim_eps)

        # a little wasteful to deepcopy (e.g. duplicates action selector), but should work for any MAC
        self.target_Genome = copy.deepcopy(genome)
        self.log_stats_t = -self.args.learner_log_interval - 1
        self.train_t = 0

        # priority replay
        self.use_per = getattr(self.args, 'use_per', False)
        self.return_priority = getattr(self.args, "return_priority", False)
        if self.use_per:
            self.priority_max = float('-inf')
            self.priority_min = float('inf')
            
        # Evolution: Unified Genome population
        # Each Genome contains: mac1 + mac2 + mixer
        self.pop_size = args.pop_size
        self.elite_size = args.elite_size
        
        self.pop_genome = pop_genome  # Population of Genomes (包含 MAC1 + MAC2 + Mixer)
        
        self.best_agents = list(range(int(self.pop_size*self.elite_size)))
        
        # Elite Archive for Novelty Search (Quality-Diversity)
        self.elite_archive = []  # Stores behavioral descriptors of historical elites
        self.archive_max_size = getattr(args, 'archive_max_size', 50)  # Maximum archive size
        self.novelty_k_nearest = getattr(args, 'novelty_k_nearest', 5)  # K-nearest neighbors for novelty
        

    def get_params(self):
        """
        Returns the total number of parameters in the model as an integer
        """
        return int(sum(p.numel() for p in self.params if hasattr(p, 'numel')))
    
    
    def calculate_TD_error(self, batch: EpisodeBatch, genome_index: int):
        """
        Calculate TD error for a specific Genome in the population.
        Uses pop_genome[genome_index] for OPTIMALITY evaluation.
        
        Args:
            batch: EpisodeBatch containing experience data
            genome_index: Index of the Genome in self.pop_genome to evaluate
            
        Returns:
            mean_td_error: Mean TD error for the specified Genome (scalar tensor)
        """
        # Get the Genome from population (contains mac1 and mac2)
        pop_genome = self.pop_genome[genome_index]
        
        # Get the relevant quantities (same as in train())
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Set both agents to eval mode for evaluation
        pop_genome.mac1.agent.eval()
        pop_genome.mac2.agent.eval()
        
        with th.no_grad():  # No gradients needed for evaluation
            # Calculate Q-Values using mac1
            mac_out_1 = []
            pop_genome.mac1.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = pop_genome.mac1.forward(batch, t=t)
                mac_out_1.append(agent_outs)
            mac_out_1 = th.stack(mac_out_1, dim=1)
            
            # Calculate Q-Values using mac2
            mac_out_2 = []
            pop_genome.mac2.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = pop_genome.mac2.forward(batch, t=t)
                mac_out_2.append(agent_outs)
            mac_out_2 = th.stack(mac_out_2, dim=1)
        
        # Pick the Q-Values for the actions taken - for both MACs
        chosen_action_qvals_1 = th.gather(mac_out_1[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim=3, index=actions).squeeze(3)
        
        # Calculate the Q-Values necessary for the target 
        with th.no_grad():
            self.target_Genome.mac1.agent.eval()
            self.target_Genome.mac2.agent.eval()
            
            # Target Q-values from target_Genome
            target_mac_out = []
            self.target_Genome.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_Genome.forward(batch, t=t)  # min(q1, q2)
                target_mac_out.append(target_agent_outs)

            target_mac_out = th.stack(target_mac_out, dim=1)

            # Max over target Q-Values/ Double q learning 
            # Use the minimum of mac1 and mac2 for action selection
            mac_out_combined = th.min(mac_out_1, mac_out_2)
            mac_out_detach = mac_out_combined.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets 
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"])

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else:
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer - for both MACs
        # Use the mixer from the population genome being evaluated
        with th.no_grad():
            chosen_action_qvals_1 = self.mixer(chosen_action_qvals_1, batch["state"][:, :-1])
            chosen_action_qvals_2 = self.mixer(chosen_action_qvals_2, batch["state"][:, :-1])

        # Calculate TD error for both MACs and sum them
        td_error_1 = (chosen_action_qvals_1 - targets.detach())
        td_error2_1 = 0.5 * td_error_1.pow(2)
        
        td_error_2 = (chosen_action_qvals_2 - targets.detach())
        td_error2_2 = 0.5 * td_error_2.pow(2)

        mask_expanded = mask.expand_as(td_error2_1)
        masked_td_error_1 = td_error2_1 * mask_expanded
        masked_td_error_2 = td_error2_2 * mask_expanded

        # Average TD errors from both MACs (not sum!)
        mean_td_error_1 = masked_td_error_1.sum() / (mask_expanded.sum() + 1e-8)
        mean_td_error_2 = masked_td_error_2.sum() / (mask_expanded.sum() + 1e-8)
        mean_td_error = (mean_td_error_1 + mean_td_error_2) / 2.0  # Take average of both MACs
        
        # Safety check: if TD error is unreasonably large, return infinity (mark as invalid)
        if mean_td_error.item() > 1e6:
            return th.tensor(1e6, device=mean_td_error.device)  # Cap at 1 million
        
        return mean_td_error


    def calculate_confidence_Q(self, batch: EpisodeBatch, genome_index: int):
        """
        Calculate confidence Q metric for a specific Genome in the population.
        Uses pop_genome[genome_index] for OPTIMALITY evaluation.
        
        Args:
            batch: EpisodeBatch containing experience data
            genome_index: Index of the Genome in self.pop_genome to evaluate
            
        Returns:
            mean_confidence_Q: Mean confidence Q for the specified Genome (scalar tensor)
        """
        # Get the Genome from population
        pop_genome = self.pop_genome[genome_index]
        
        # Get the relevant quantities (same as in train())
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values using the population Genome
        pop_genome.eval()  # Set to eval mode for evaluation
        pop_genome.init_hidden(batch.batch_size)
        mac_out = []
        
        with th.no_grad():  # No gradients needed for evaluation
            for t in range(batch.max_seq_length):
                agent_outs = pop_genome.forward(batch, t=t)  # min(q1, q2)
                mac_out.append(agent_outs)
        
        mac_out = th.stack(mac_out, dim=1)  # (batch, seq_len, n_agents, n_actions)
        
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        # Calculate mean confidence Q using the mixer from the population genome
        with th.no_grad():
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        mean_confidence_Q = (chosen_action_qvals * mask).sum() / mask.sum()

        return mean_confidence_Q


    def calculate_adversarial_entropy(self, batch, genome_index):
        """
        Calculate adversarial entropy for robustness evaluation.
        Uses pop_genome[genome_index] for ROBUSTNESS evaluation.
        
        Args:
            batch: EpisodeBatch containing experience data
            genome_index: Index of the Genome in self.pop_genome to evaluate
            
        Returns:
            mean_entropy: Entropy (higher = more uncertain/less confident)
        """
        # Get the Genome from population
        pop_genome = self.pop_genome[genome_index]
        eval_mixer = self.mixer
        
        # Get mask for valid timesteps
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # Set to eval mode
        pop_genome.eval()
        
        # Create adversarial observations
        adv_batch = noise_atk(batch, self.args)
        
        # Compute adversarial Global Q-values for all actions
        pop_genome.init_hidden(batch.batch_size)
        adv_global_q_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                # Get Q-values for all actions from pop_genome (min of mac1 and mac2)
                agent_out = pop_genome.forward(adv_batch, t=t)  # (batch, n_agents, n_actions)
                adv_global_q_list.append(agent_out)
        
        # Stack: (batch, seq_len, n_agents, n_actions)
        adv_agent_q = th.stack(adv_global_q_list, dim=1)
        
        # Apply EVOLVED mixer to get Global Q for all actions
        with th.no_grad():
            adv_global_q = eval_mixer(adv_agent_q, adv_batch["state"])[:, :-1]  # (batch, seq_len-1, n_actions)
        
        # Compute Boltzmann distribution (softmax with temperature)
        tau = getattr(self.args, 'adversarial_tau', 1.0)  # Temperature parameter
        
        # Compute probabilities: π(a|o_adv) = softmax(Q_tot(o_adv, a) / τ)
        log_probs = th.nn.functional.log_softmax(adv_global_q / tau, dim=-1)  # (batch, seq_len-1, n_actions)
        probs = th.exp(log_probs)  # (batch, seq_len-1, n_actions)
        
        # Compute entropy: H(π) = -Σ π(a) log π(a)
        entropy = -(probs * log_probs).sum(dim=-1)  # (batch, seq_len-1)
        
        # Apply mask and compute mean
        masked_entropy = entropy * mask.unsqueeze(-1)
        mean_entropy = masked_entropy.sum() / mask.sum()
        
        # Return entropy directly (caller will negate it for "confidence" interpretation)
        return mean_entropy
    
    def calculate_adversarial_loss(self, batch, genome_index):
        """
        Fitness 3: 故障隔离率 (Fault Isolation Ratio)
        
        计算 Mixer 对拜占庭故障 Agent 的隔离能力。
        场景: 假设 Agent k 发生拜占庭故障（输出最差/随机 Q 值）
        
        数学定义: 负的 Global/Local 敏感度比率
        F3 = -E[|Q_tot(u_normal) - Q_tot(u_faulty)| / (|Q_k - Q'_k| + ε)]
        
        物理含义:
        - 分母: 故障的严重程度（Local Q 的变化）
        - 分子: Global Q 受到的波及程度
        - 比率越小 => Mixer 成功隔离故障影响（W_k → 0）
        
        Args:
            batch: EpisodeBatch containing experience data
            genome_index: Index of the Genome in self.pop_genome to evaluate
            
        Returns:
            mean_isolation_ratio: 负的平均隔离比率（越大越好）
        """
        # Get the Genome from population
        pop_genome = self.pop_genome[genome_index]
        
        # Get mask for valid timesteps
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = batch["actions"][:, :-1]
        
        # Set to eval mode
        pop_genome.eval()
        
        # === Step 1: 计算正常情况下的 Q 值 ===
        pop_genome.init_hidden(batch.batch_size)
        normal_agent_q_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                agent_out = pop_genome.forward(batch, t=t)  # min(q1, q2)
                normal_agent_q_list.append(agent_out)
        
        normal_agent_q = th.stack(normal_agent_q_list, dim=1)  # (batch, seq_len, n_agents, n_actions)
        normal_chosen_q = th.gather(normal_agent_q[:, :-1], dim=3, index=actions).squeeze(3)  # (batch, seq_len-1, n_agents)
        
        # 计算正常的 Global Q
        with th.no_grad():
            normal_global_q = self.mixer(normal_chosen_q, batch["state"][:, :-1])  # (batch, seq_len-1, 1)
        
        # === Step 2: 使用攻击者选择的 Agent 进行故障注入 ===
        n_agents = self.args.n_agents
        batch_size = batch.batch_size
        seq_len = normal_chosen_q.shape[1]
        
        # 使用batch中记录的攻击者选择的受害者 (由Attacker网络选择)
        # victim_id shape: (batch, seq_len, 1), 值为 [0, n_agents], n_agents表示"不攻击"
        victim_ids = batch["victim_id"][:, :-1].squeeze(-1)  # (batch, seq_len-1)
        
        # 对每个样本,选择该样本中最常被攻击的Agent作为故障Agent
        # 如果全是no-attack(n_agents),则随机选择
        faulty_agent_indices = th.zeros(batch_size, dtype=th.long, device=self.device)
        for b in range(batch_size):
            victim_counts = th.bincount(victim_ids[b].long(), minlength=n_agents+1)[:n_agents]
            if victim_counts.sum() > 0:  # 有实际攻击发生
                faulty_agent_indices[b] = victim_counts.argmax()
            else:  # 该样本中没有攻击,随机选择
                faulty_agent_indices[b] = th.randint(0, n_agents, (1,), device=self.device)
        
        # === Step 3: 注入拜占庭故障 ===
        faulty_chosen_q = normal_chosen_q.clone()  # (batch, seq_len-1, n_agents)
        
        # 故障类型: 随机扰动或设为最差值
        fault_type = getattr(self.args, 'fault_type', 'worst')
        
        for b in range(batch_size):
            k = faulty_agent_indices[b].item()
            
            if fault_type == 'worst':
                # 设置为该 batch 该 agent 的最小 Q 值（最差动作）
                min_q = normal_chosen_q[b, :, k].min()
                faulty_chosen_q[b, :, k] = min_q
            elif fault_type == 'random':
                # 随机噪声
                noise = th.randn_like(faulty_chosen_q[b, :, k]) * 0.5
                faulty_chosen_q[b, :, k] = normal_chosen_q[b, :, k] + noise
            else:  # 'zero'
                faulty_chosen_q[b, :, k] = 0.0
        
        # 计算故障情况下的 Global Q
        with th.no_grad():
            faulty_global_q = self.mixer(faulty_chosen_q, batch["state"][:, :-1])  # (batch, seq_len-1, 1)
        
        # === Step 4: 计算隔离比率 ===
        # 分子: Global Q 的变化（波及程度）
        global_delta = th.abs(normal_global_q - faulty_global_q)  # (batch, seq_len-1, 1)
        
        # 分母: Local Q 的变化（故障严重程度）
        local_delta_list = []
        for b in range(batch_size):
            k = faulty_agent_indices[b].item()
            local_change = th.abs(normal_chosen_q[b, :, k] - faulty_chosen_q[b, :, k])  # (seq_len-1,)
            local_delta_list.append(local_change)
        
        local_delta = th.stack(local_delta_list, dim=0).unsqueeze(-1)  # (batch, seq_len-1, 1)
        
        # 隔离比率 = Global 变化 / Local 变化
        # 加 epsilon 防止除零
        epsilon = 1e-6
        isolation_ratio = global_delta / (local_delta + epsilon)  # (batch, seq_len-1, 1)
        
        # 应用 mask 并计算平均
        masked_ratio = isolation_ratio * mask
        mean_isolation_ratio = masked_ratio.sum() / (mask.sum() + 1e-8)
        
        # 返回负值（比率越小，fitness 越高）
        return -mean_isolation_ratio
    
    def calculate_influence_constraint(self, batch, genome_index):
        """
        Fitness 4: 权力制衡约束 (Influence Constraint)
        
        计算 Mixer 权重分配的去中心化程度，防止"独裁结构"。
        场景: 不看具体故障，只看权重分配结构
        
        数学定义: 负的最大单体影响力
        首先计算每个 Agent 的影响力: I_k(s) = |Q_tot(Q_k+δ) - Q_tot(Q_k)|
        F4 = -E[max_k I_k(s) + λ·Var(I)]
        
        物理含义:
        - max_k I_k: 最大单体影响力（防止某个 Agent 权重过大）
        - Var(I): 影响力方差（鼓励均衡分配）
        - 迫使 Mixer 进化出去中心化、冗余的权重分配机制
        
        Args:
            batch: EpisodeBatch containing experience data
            genome_index: Index of the Genome in self.pop_genome to evaluate
            
        Returns:
            constraint_score: 负的影响力约束值（越大越好）
        """
        # Get the Genome from population
        pop_genome = self.pop_genome[genome_index]
        
        # Get mask for valid timesteps
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = batch["actions"][:, :-1]
        
        # Set to eval mode
        pop_genome.eval()
        
        # === Step 1: 计算基准 Q 值 ===
        pop_genome.init_hidden(batch.batch_size)
        agent_q_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                agent_out = pop_genome.forward(batch, t=t)  # min(q1, q2)
                agent_q_list.append(agent_out)
        
        agent_q = th.stack(agent_q_list, dim=1)  # (batch, seq_len, n_agents, n_actions)
        chosen_q = th.gather(agent_q[:, :-1], dim=3, index=actions).squeeze(3)  # (batch, seq_len-1, n_agents)
        
        # 基准 Global Q
        with th.no_grad():
            baseline_global_q = self.mixer(chosen_q, batch["state"][:, :-1])  # (batch, seq_len-1, 1)
        
        # === Step 2: 计算每个 Agent 的影响力 ===
        n_agents = self.args.n_agents
        delta = getattr(self.args, 'influence_delta', 0.1)  # 扰动幅度
        
        influences = []  # List of (batch, seq_len-1) tensors
        
        for k in range(n_agents):
            # 对 Agent k 的 Q 值添加扰动
            perturbed_q = chosen_q.clone()  # (batch, seq_len-1, n_agents)
            perturbed_q[:, :, k] = perturbed_q[:, :, k] + delta
            
            # 计算扰动后的 Global Q
            with th.no_grad():
                perturbed_global_q = self.mixer(perturbed_q, batch["state"][:, :-1])  # (batch, seq_len-1, 1)
            
            # 影响力 = |Q_tot(Q_k+δ) - Q_tot(Q_k)|
            influence_k = th.abs(perturbed_global_q - baseline_global_q).squeeze(-1)  # (batch, seq_len-1)
            influences.append(influence_k)
        
        # Stack: (n_agents, batch, seq_len-1)
        influences_tensor = th.stack(influences, dim=0)
        
        # === Step 3: 计算最大影响力和方差 ===
        # 最大单体影响力
        max_influence, _ = influences_tensor.max(dim=0)  # (batch, seq_len-1)
        
        # 影响力方差（跨 agents）
        influence_var = influences_tensor.var(dim=0)  # (batch, seq_len-1)
        
        # === Step 4: 计算约束值 ===
        lambda_var = getattr(self.args, 'influence_lambda', 0.5)  # 方差权重
        
        constraint_per_timestep = max_influence + lambda_var * influence_var  # (batch, seq_len-1)
        
        # 应用 mask 并计算平均
        masked_constraint = constraint_per_timestep * mask
        mean_constraint = masked_constraint.sum() / (mask.sum() + 1e-8)
        
        # 返回负值（约束值越小，fitness 越高）
        return -mean_constraint
    
    def calculate_evolutionary_consensus(self, batch, genome_index, elite_indices):
        """
        机制一：基于种群共识的鲁棒性校准 (Population-Consensus Calibration)
        
        **痛点**: 单个 genome 在面对故障时可能会产生"幻觉"（过度悲观或乐观）
        **解决方案**: 利用群体智慧 (Swarm Intelligence) 校准个体的鲁棒性判断
        
        在故障场景下，要求当前 Genome 的 Q_tot 判断与种群 Top-K 精英的平均判断保持一致：
        Penalty_consensus = ||Q_tot^me(faulty) - Mean(Q_tot^elites(faulty))||
        
        Args:
            batch: EpisodeBatch containing experience data
            genome_index: 当前评估的 Genome 索引
            elite_indices: 精英 Genome 索引列表（按 fitness 排序）
            
        Returns:
            consensus_penalty: 负的偏离度（越高=越一致，fitness 越好）
        """
        # 获取当前评估的 Genome
        pop_genome = self.pop_genome[genome_index]
        
        # 获取 mask
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = batch["actions"][:, :-1]
        
        # 设置为 eval 模式
        pop_genome.eval()
        
        # === Step 1: 创建故障场景（使用攻击者选择的Agent）===
        n_agents = self.args.n_agents
        batch_size = batch.batch_size
        
        # 使用batch中记录的攻击者选择的受害者 (由Attacker网络选择)
        victim_ids = batch["victim_id"][:, :-1].squeeze(-1)  # (batch, seq_len-1)
        
        # 对每个样本,选择该样本中最常被攻击的Agent作为故障Agent
        faulty_agent_indices = th.zeros(batch_size, dtype=th.long, device=self.device)
        for b in range(batch_size):
            victim_counts = th.bincount(victim_ids[b].long(), minlength=n_agents+1)[:n_agents]
            if victim_counts.sum() > 0:  # 有实际攻击发生
                faulty_agent_indices[b] = victim_counts.argmax()
            else:  # 该样本中没有攻击,随机选择
                faulty_agent_indices[b] = th.randint(0, n_agents, (1,), device=self.device)
        
        # 计算正常情况下的 Agent Q 值
        pop_genome.init_hidden(batch.batch_size)
        normal_agent_q_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                agent_out = pop_genome.forward(batch, t=t)  # min(q1, q2)
                normal_agent_q_list.append(agent_out)
        
        normal_agent_q = th.stack(normal_agent_q_list, dim=1)
        normal_chosen_q = th.gather(normal_agent_q[:, :-1], dim=3, index=actions).squeeze(3)
        
        # 注入故障
        faulty_chosen_q = normal_chosen_q.clone()
        fault_type = getattr(self.args, 'consensus_fault_type', 'worst')
        
        for b in range(batch_size):
            k = faulty_agent_indices[b].item()
            if fault_type == 'worst':
                min_q = normal_chosen_q[b, :, k].min()
                faulty_chosen_q[b, :, k] = min_q
            elif fault_type == 'random':
                noise = th.randn_like(faulty_chosen_q[b, :, k]) * 0.5
                faulty_chosen_q[b, :, k] = normal_chosen_q[b, :, k] + noise
            else:  # 'zero'
                faulty_chosen_q[b, :, k] = 0.0
        
        # === Step 2: 计算当前 Genome 在故障场景下的 Q_tot ===
        with th.no_grad():
            my_faulty_global_q = self.mixer(faulty_chosen_q, batch["state"][:, :-1])
        
        # === Step 3: 计算精英种群在相同故障场景下的平均 Q_tot ===
        K = min(len(elite_indices), getattr(self.args, 'consensus_elite_size', 3))
        
        if K == 0 or genome_index in elite_indices[:K]:
            # 如果没有精英或自己就是精英，返回 0（无惩罚）
            return th.tensor(0.0, device=self.device)
        
        elite_faulty_global_qs = []
        for elite_idx in elite_indices[:K]:
            if elite_idx == genome_index:
                continue
            
            elite_genome = self.pop_genome[elite_idx]
            elite_genome.eval()
            
            with th.no_grad():
                # 使用相同的故障场景评估精英
                elite_faulty_global_q = self.mixer(faulty_chosen_q, batch["state"][:, :-1])
            elite_faulty_global_qs.append(elite_faulty_global_q)
        
        if len(elite_faulty_global_qs) == 0:
            return th.tensor(0.0, device=self.device)
        
        # 计算精英平均判断（群体智慧）
        elite_avg_faulty_q = th.stack(elite_faulty_global_qs, dim=0).mean(dim=0)
        
        # === Step 4: 计算共识偏离度 ===
        deviation = th.abs(my_faulty_global_q - elite_avg_faulty_q)
        masked_deviation = deviation * mask
        mean_deviation = masked_deviation.sum() / (mask.sum() + 1e-8)
        
        # 返回负值（偏离越小，fitness 越高）
        # TEVC 亮点：利用 Swarm Intelligence 来校准个体的鲁棒性判断
        return -mean_deviation
    
    def calculate_adversarial_novelty(self, batch, genome_index):
        """
        机制三：基于权重分布的新颖性搜索 (Weighting-Pattern Novelty)
        
        **痛点**: 所有 Mixer 可能收敛到同一种平庸的防御策略（如简单平均）
        **解决方案**: Quality-Diversity (QD) - 鼓励探索不同的权力分配结构
        
        提取 Mixer 的权重分配模式作为行为描述符，计算与历史精英存档的距离。
        有的 Mixer 可能集权防御（某些 Agent 权重大），有的分权防御（均衡分配）。
        
        行为描述符: Mixer 在一批数据上产生的影响力分布向量 I ∈ R^N
        新颖性得分: avg_distance(I_me, I_archive)
        
        Args:
            batch: EpisodeBatch containing experience data
            genome_index: 当前评估的 Genome 索引
            
        Returns:
            novelty_score: 与存档的平均距离（越高=越新颖，fitness 越好）
        """
        # 获取当前评估的 Genome
        pop_genome = self.pop_genome[genome_index]
        
        # 获取 mask
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = batch["actions"][:, :-1]
        
        # 设置为 eval 模式
        pop_genome.eval()
        
        # === Step 1: 计算基准 Agent Q 值 ===
        pop_genome.init_hidden(batch.batch_size)
        agent_q_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                agent_out = pop_genome.forward(batch, t=t)
                agent_q_list.append(agent_out)
        
        agent_q = th.stack(agent_q_list, dim=1)
        chosen_q = th.gather(agent_q[:, :-1], dim=3, index=actions).squeeze(3)
        
        # 基准 Global Q
        with th.no_grad():
            baseline_global_q = self.mixer(chosen_q, batch["state"][:, :-1])
        
        # === Step 2: 提取权重分配模式（影响力分布）作为行为描述符 ===
        n_agents = self.args.n_agents
        delta = getattr(self.args, 'novelty_delta', 0.1)
        
        influences = []
        
        for k in range(n_agents):
            # 对 Agent k 的 Q 值添加扰动
            perturbed_q = chosen_q.clone()
            perturbed_q[:, :, k] = perturbed_q[:, :, k] + delta
            
            # 计算扰动后的 Global Q
            with th.no_grad():
                perturbed_global_q = self.mixer(perturbed_q, batch["state"][:, :-1])
            
            # 影响力 I_k = |Q_tot(Q_k+δ) - Q_tot(Q_k)|
            influence_k = th.abs(perturbed_global_q - baseline_global_q).squeeze(-1)
            influences.append(influence_k)
        
        # Stack: (n_agents, batch, seq_len-1) -> transpose to (batch, seq_len-1, n_agents)
        influences_tensor = th.stack(influences, dim=0)  # (n_agents, batch, seq_len-1)
        influences_tensor = influences_tensor.permute(1, 2, 0)  # (batch, seq_len-1, n_agents)
        
        # 对时间和 batch 维度取平均，得到行为描述符
        # behavior_descriptor: (n_agents,) - 每个 Agent 的平均影响力
        masked_influences = influences_tensor * mask  # (batch, seq_len-1, n_agents) * (batch, seq_len-1, 1)
        behavior_me = masked_influences.sum(dim=(0, 1)) / (mask.sum() + 1e-8)  # Sum over batch and time
        
        # 归一化为概率分布（权重分配模式）
        behavior_me = behavior_me / (behavior_me.sum() + 1e-8)
        
        # === Step 3: 计算与存档的距离（新颖性） ===
        if len(self.elite_archive) == 0:
            # 空存档：最大新颖度
            return th.tensor(1.0, device=self.device)
        
        distances = []
        for archived_behavior in self.elite_archive:
            # 使用欧氏距离度量权重分配模式的差异
            dist = th.norm(behavior_me - archived_behavior, p=2)
            distances.append(dist)
        
        distances_tensor = th.stack(distances, dim=0)
        
        # K 近邻距离
        K = min(self.novelty_k_nearest, len(self.elite_archive))
        knn_distances, _ = th.topk(distances_tensor, K, largest=False)
        
        # 新颖度 = K 近邻的平均距离
        novelty_score = knn_distances.mean()
        
        # TEVC 亮点：Quality-Diversity - 防止种群陷入局部最优
        # 鼓励 Mixer 探索不同的权力分配结构（集权 vs 分权）
        return novelty_score
    
    def _js_divergence(self, p, q):
        """
        Compute Jensen-Shannon divergence between two distributions.
        JS(P||Q) = 0.5 * KL(P||M) + 0.5 * KL(Q||M), where M = (P+Q)/2
        
        Args:
            p, q: Probability distributions, shape (batch, n_actions)
            
        Returns:
            js_div: JS divergence for each batch element, shape (batch,)
        """
        # Ensure valid probabilities
        p = p + 1e-8
        q = q + 1e-8
        p = p / p.sum(dim=-1, keepdim=True)
        q = q / q.sum(dim=-1, keepdim=True)
        
        # Mixture distribution
        m = 0.5 * (p + q)
        
        # KL divergences
        kl_pm = (p * (th.log(p) - th.log(m))).sum(dim=-1)
        kl_qm = (q * (th.log(q) - th.log(m))).sum(dim=-1)
        
        # JS divergence
        js_div = 0.5 * kl_pm + 0.5 * kl_qm
        
        return js_div
    
    def update_elite_archive(self, elite_indices, batch):
        """
        更新精英存档（用于新颖性搜索）
        存储 Genome 的权重分配模式（影响力分布）作为行为特征
        
        Args:
            elite_indices: 精英 Genome 索引列表
            batch: EpisodeBatch 用于计算行为描述符
        """
        # 获取 mask
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        actions = batch["actions"][:, :-1]
        
        n_agents = self.args.n_agents
        delta = getattr(self.args, 'novelty_delta', 0.1)
        
        # 为每个精英 Genome 计算行为描述符（权重分配模式）
        for elite_idx in elite_indices[:min(len(elite_indices), 5)]:  # Top-5 精英
            elite_genome = self.pop_genome[elite_idx]
            elite_genome.eval()
            
            # 计算 Agent Q 值
            elite_genome.init_hidden(batch.batch_size)
            agent_q_list = []
            
            with th.no_grad():
                for t in range(batch.max_seq_length):
                    agent_out = elite_genome.forward(batch, t=t)
                    agent_q_list.append(agent_out)
            
            agent_q = th.stack(agent_q_list, dim=1)
            chosen_q = th.gather(agent_q[:, :-1], dim=3, index=actions).squeeze(3)
            
            # 基准 Global Q
            with th.no_grad():
                baseline_global_q = self.mixer(chosen_q, batch["state"][:, :-1])
            
            # 计算每个 Agent 的影响力
            influences = []
            for k in range(n_agents):
                perturbed_q = chosen_q.clone()
                perturbed_q[:, :, k] = perturbed_q[:, :, k] + delta
                
                with th.no_grad():
                    perturbed_global_q = self.mixer(perturbed_q, batch["state"][:, :-1])
                
                influence_k = th.abs(perturbed_global_q - baseline_global_q).squeeze(-1)
                influences.append(influence_k)
            
            influences_tensor = th.stack(influences, dim=0)  # (n_agents, batch, seq_len-1)
            influences_tensor = influences_tensor.permute(1, 2, 0)  # (batch, seq_len-1, n_agents)
            
            # 行为描述符：每个 Agent 的平均影响力（权重分配模式）
            masked_influences = influences_tensor * mask  # (batch, seq_len-1, n_agents) * (batch, seq_len-1, 1)
            behavior = masked_influences.sum(dim=(0, 1)) / (mask.sum() + 1e-8)  # Sum over batch and time
            
            # 归一化为概率分布
            behavior = behavior / (behavior.sum() + 1e-8)
            
            # 添加到存档
            self.elite_archive.append(behavior.detach().clone())
        
        # 维护存档大小（FIFO）
        if len(self.elite_archive) > self.archive_max_size:
            self.elite_archive = self.elite_archive[-self.archive_max_size:]
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] # the total reward
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) # whether the sample is valid
        avail_actions = batch["avail_actions"]
                
        
        # Calculate estimated Q-Values for BOTH MACs in Genome
        self.Genome.train()  # Set both mac1 and mac2 to train mode
        
        # Forward pass for mac1
        mac_out_1 = []
        self.Genome.mac1.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            agent_outs_1 = self.Genome.mac1.forward(batch, t=t)
            mac_out_1.append(agent_outs_1)
            
        # Forward pass for mac2
        mac_out_2 = []
        self.Genome.mac2.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            agent_outs_2 = self.Genome.mac2.forward(batch, t=t)
            mac_out_2.append(agent_outs_2)
            
        mac_out_1 = th.stack(mac_out_1, dim=1)  # (batch, seq_len, n_agents, n_actions)
        mac_out_2 = th.stack(mac_out_2, dim=1)  # (batch, seq_len, n_agents, n_actions)
        
        # Pick the Q-Values for the actions taken by each agent - for both MACs
        chosen_action_qvals_1 = th.gather(mac_out_1[:, :-1], dim=3, index=actions).squeeze(3)
        chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_Genome.train()  # Set target Genome to train mode
            target_mac_out = []
            self.target_Genome.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_Genome.forward(batch, t=t)  # min(q1, q2)
                target_mac_out.append(target_agent_outs)

            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values/ Double q learning
            # Use min(mac1, mac2) for action selection
            mac_out_combined = th.min(mac_out_1, mac_out_2)
            mac_out_detach = mac_out_combined.clone().detach()
            mac_out_detach[avail_actions == 0] = -9999999
            cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
            target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
            
            # Calculate n-step Q-Learning targets
            target_max_qvals = self.target_mixer(target_max_qvals, batch["state"]) # (batch, seq_len, 1)

            if getattr(self.args, 'q_lambda', False):
                qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                qvals = self.target_mixer(qvals, batch["state"])

                targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                    self.args.gamma, self.args.td_lambda)
            else: # (batch, seq_len - 1, n_agents) the last timestep is removed
                targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals, 
                                                    self.args.n_agents, self.args.gamma, self.args.td_lambda)

        # Mixer - for both MACs
        chosen_action_qvals_1 = self.mixer(chosen_action_qvals_1, batch["state"][:, :-1])
        chosen_action_qvals_2 = self.mixer(chosen_action_qvals_2, batch["state"][:, :-1])

        # Calculate TD error for BOTH MACs
        td_error_1 = (chosen_action_qvals_1 - targets.detach())
        td_error2_1 = 0.5 * td_error_1.pow(2)
        
        td_error_2 = (chosen_action_qvals_2 - targets.detach())
        td_error2_2 = 0.5 * td_error_2.pow(2)

        mask = mask.expand_as(td_error2_1)
        masked_td_error_1 = td_error2_1 * mask
        masked_td_error_2 = td_error2_2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error_1 = masked_td_error_1.sum(1) * per_weight
            masked_td_error_2 = masked_td_error_2.sum(1) * per_weight

        # Sum TD errors from both MACs - this is the key change!
        L_td_1 = masked_td_error_1.sum() / mask.sum()
        L_td_2 = masked_td_error_2.sum() / mask.sum()
        L_td = L_td_1 + L_td_2  # Total loss is sum of both MACs
        loss = L_td
        
        # Optimise
        self.optimiser.zero_grad()
        loss.backward()
        grad_norm = th.nn.utils.clip_grad_norm_(self.params, self.args.grad_norm_clip)
        self.optimiser.step()

        if (episode_num - self.last_target_update_episode) / self.args.target_update_interval >= 1.0:
            self._update_targets()
            self.last_target_update_episode = episode_num

        if t_env - self.log_stats_t >= self.args.learner_log_interval:
            self.logger.log_stat("loss_td", L_td.item(), t_env)
            self.logger.log_stat("grad_norm", grad_norm, t_env)
            mask_elems = mask.sum().item()
            # Use combined masked_td_error for logging
            masked_td_error_combined = masked_td_error_1 + masked_td_error_2
            self.logger.log_stat("td_error_abs", (masked_td_error_combined.abs().sum().item()/mask_elems), t_env)
            # Use average of both MACs for q_taken_mean
            chosen_action_qvals_avg = (chosen_action_qvals_1 + chosen_action_qvals_2) / 2.0
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals_avg * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                # Use combined mac_out for matrix status
                mac_out_combined = th.min(mac_out_1, mac_out_2)
                print_matrix_status(batch, self.mixer, mac_out_combined)

        # return info
        info = {}
        # calculate priority
        if self.use_per:
            if self.return_priority:
                info["td_errors_abs"] = rewards.sum(1).detach().to('cpu')
                # normalize to [0, 1]
                self.priority_max = max(th.max(info["td_errors_abs"]).item(), self.priority_max)
                self.priority_min = min(th.min(info["td_errors_abs"]).item(), self.priority_min)
                info["td_errors_abs"] = (info["td_errors_abs"] - self.priority_min) \
                                / (self.priority_max - self.priority_min + 1e-5)
            else:
                # Use combined td_error from both MACs
                td_error_combined = td_error_1 + td_error_2
                info["td_errors_abs"] = ((td_error_combined.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        
        # Store TD error for learning-assisted dynamic weighting
        self.last_td_error = L_td.item()
        info["td_error"] = self.last_td_error
        
        return info
    
    def lamarckian_finetune(self, pop_genome, batch, num_sgd_steps=1):
        """
        Lamarckian SGD Injection: 对种群中的单个 Genome 进行 SGD 微调。
        
        **Lamarckian 特性**: 微调直接修改个体的权重（基因），因此获得的改进会遗传给后代。
        与达尔文进化不同，这里"后天学习"的结果会改变遗传信息。
        
        Args:
            pop_genome: 要微调的 Genome（来自 self.pop_genome）
            batch: EpisodeBatch 用于 SGD 训练
            num_sgd_steps: SGD 迭代次数（默认 1）
            
        Returns:
            td_loss: 微调后的 TD 损失（用于监控）
        """
        # 1. 为该个体创建临时优化器（避免影响主 RL 优化器）
        genome_params = list(pop_genome.parameters())
        
        # 使用更保守的学习率进行微调（防止破坏进化得到的结构和梯度爆炸）
        finetune_lr = getattr(self.args, 'finetune_lr', self.args.lr * 0.01)  # 改为 0.01 (更保守)
        
        if self.args.optimizer == 'adam':
            genome_optimizer = Adam(params=genome_params, lr=finetune_lr)
        else:
            genome_optimizer = RMSprop(params=genome_params, lr=finetune_lr, 
                                      alpha=self.args.optim_alpha, eps=self.args.optim_eps)
        
        # 初始化 loss（防止循环内跳过导致未定义）
        L_td = th.tensor(0.0, device=self.device)
        
        # 2. 执行 SGD 步骤
        for _ in range(num_sgd_steps):
            # Get the relevant quantities
            rewards = batch["reward"][:, :-1]
            actions = batch["actions"][:, :-1]
            terminated = batch["terminated"][:, :-1].float()
            mask = batch["filled"][:, :-1].float()
            mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
            avail_actions = batch["avail_actions"]
            
            # Set to train mode
            pop_genome.train()
            
            # Forward pass for mac1
            mac_out_1 = []
            pop_genome.mac1.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs_1 = pop_genome.mac1.forward(batch, t=t)
                mac_out_1.append(agent_outs_1)
            
            # Forward pass for mac2
            mac_out_2 = []
            pop_genome.mac2.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs_2 = pop_genome.mac2.forward(batch, t=t)
                mac_out_2.append(agent_outs_2)
            
            mac_out_1 = th.stack(mac_out_1, dim=1)
            mac_out_2 = th.stack(mac_out_2, dim=1)
            
            # Pick Q-values for actions taken
            chosen_action_qvals_1 = th.gather(mac_out_1[:, :-1], dim=3, index=actions).squeeze(3)
            chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim=3, index=actions).squeeze(3)
            
            # Calculate targets using pop_genome itself (detached to avoid gradients)
            # 关键修复：不使用 self.target_Genome，而是用 pop_genome 自己的 detached 版本
            with th.no_grad():
                # 使用 pop_genome 自己来计算目标（而非主 RL 的 target 网络）
                pop_genome.eval()  # 临时设为 eval 模式计算目标
                
                # 计算目标 Q 值
                target_mac_out_1 = []
                pop_genome.mac1.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_agent_outs = pop_genome.mac1.forward(batch, t=t)
                    target_mac_out_1.append(target_agent_outs)
                
                target_mac_out_2 = []
                pop_genome.mac2.init_hidden(batch.batch_size)
                for t in range(batch.max_seq_length):
                    target_agent_outs = pop_genome.mac2.forward(batch, t=t)
                    target_mac_out_2.append(target_agent_outs)
                
                target_mac_out_1 = th.stack(target_mac_out_1, dim=1)
                target_mac_out_2 = th.stack(target_mac_out_2, dim=1)
                
                # 使用 min(Q1, Q2) 作为目标（Double Q-learning 的保守估计）
                target_mac_out = th.min(target_mac_out_1, target_mac_out_2)
                
                # Double Q-learning: 使用当前网络选择动作，目标网络评估
                mac_out_combined = th.min(mac_out_1, mac_out_2)
                mac_out_detach = mac_out_combined.clone().detach()
                mac_out_detach[avail_actions == 0] = -9999999
                cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
                target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
                
                # 使用 pop_genome 的 mixer（而非 self.target_mixer）
                target_max_qvals = self.mixer(target_max_qvals, batch["state"])
                
                if getattr(self.args, 'q_lambda', False):
                    qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
                    qvals = self.mixer(qvals, batch["state"])
                    targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
                                                    self.args.gamma, self.args.td_lambda)
                else:
                    targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
                                                     self.args.n_agents, self.args.gamma, self.args.td_lambda)
                
                pop_genome.train()  # 恢复训练模式
            
            # Apply mixer from pop_genome
            chosen_action_qvals_1 = self.mixer(chosen_action_qvals_1, batch["state"][:, :-1])
            chosen_action_qvals_2 = self.mixer(chosen_action_qvals_2, batch["state"][:, :-1])
            
            # Calculate TD error
            td_error_1 = (chosen_action_qvals_1 - targets.detach())
            td_error2_1 = 0.5 * td_error_1.pow(2)
            
            td_error_2 = (chosen_action_qvals_2 - targets.detach())
            td_error2_2 = 0.5 * td_error_2.pow(2)
            
            mask_expanded = mask.expand_as(td_error2_1)
            masked_td_error_1 = td_error2_1 * mask_expanded
            masked_td_error_2 = td_error2_2 * mask_expanded
            
            # Total loss
            L_td_1 = masked_td_error_1.sum() / (mask_expanded.sum() + 1e-8)
            L_td_2 = masked_td_error_2.sum() / (mask_expanded.sum() + 1e-8)
            L_td = L_td_1 + L_td_2
            
            # 检查 loss 是否有效（防止 NaN/Inf 传播）
            if not th.isfinite(L_td):
                # Loss 无效，跳过此次更新
                return float('inf')
            
            # 限制 loss 的最大值（防止梯度爆炸）
            max_loss = getattr(self.args, 'max_finetune_loss', 100.0)
            if L_td.item() > max_loss:
                # Loss 过大，使用裁剪后的 loss
                L_td = th.clamp(L_td, max=max_loss)
            
            # Optimize
            genome_optimizer.zero_grad()
            L_td.backward()
            
            # 更严格的梯度裁剪
            grad_norm_clip = getattr(self.args, 'finetune_grad_clip', self.args.grad_norm_clip * 0.5)
            grad_norm = th.nn.utils.clip_grad_norm_(genome_params, grad_norm_clip)
            
            # 检查梯度是否有效
            if not th.isfinite(grad_norm):
                # 梯度无效，跳过此次更新
                genome_optimizer.zero_grad()
                return float('inf')
            
            genome_optimizer.step()
        
        # 3. 返回最终损失（用于监控微调效果）
        return L_td.item()

    def _update_targets(self):
        self.target_Genome.load_state(self.Genome)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.Genome.cuda()
        self.target_Genome.cuda()
        # 移动种群到CUDA (每个 Genome 包含 mac1, mac2, mixer)
        for pop_genome in self.pop_genome:
            pop_genome.cuda()
        if self.mixer is not None:
            self.mixer.cuda()
            self.target_mixer.cuda()
            
    def save_models(self, path):
        self.Genome.save_models(path)
        if self.mixer is not None:
            th.save(self.mixer.state_dict(), "{}/mixer.th".format(path))
        th.save(self.optimiser.state_dict(), "{}/opt.th".format(path))

    def load_models(self, path):
        self.Genome.load_models(path)
        # Not quite right but I don't want to save target networks
        self.target_Genome.load_models(path)
        if self.mixer is not None:
            self.mixer.load_state_dict(th.load("{}/mixer.th".format(path), map_location=lambda storage, loc: storage))
        self.optimiser.load_state_dict(th.load("{}/opt.th".format(path), map_location=lambda storage, loc: storage))

    # ========== Attacker Fitness Calculation ==========
    def calculate_attacker_quality(self, batch, attacker_idx, population_attackers):
        """
        Calculate attack quality: negative mean episode return (lower defender reward = better).
        
        Args:
            batch: Episode batch data
            attacker_idx: Index of attacker in population
            population_attackers: List of all attacker instances
            
        Returns:
            mean_return: Mean episode return (scalar tensor)
        """
        # Extract rewards from batch
        rewards = batch["reward"][:, :-1]  # [batch_size, T, 1]
        mask = batch["filled"][:, :-1].float()  # [batch_size, T, 1]
        
        # Compute episode returns
        episode_returns = (rewards * mask).sum(dim=1) / mask.sum(dim=1)  # [batch_size, 1]
        mean_return = episode_returns.mean()
        
        return mean_return
    
    def calculate_attacker_efficiency(self, batch, attacker_idx, population_attackers):
        """
        Calculate attack efficiency: mean attack count (lower is more efficient).
        
        Args:
            batch: Episode batch data
            attacker_idx: Index of attacker in population
            population_attackers: List of all attacker instances
            
        Returns:
            mean_attack_count: Mean number of attacks used (scalar tensor)
        """
        # Extract attack budget usage from batch
        # Assuming "left_attack" tracks remaining budget (initial - used)
        left_attack = batch["left_attack"][:, :-1]  # [batch_size, T, 1]
        mask = batch["filled"][:, :-1].float()
        
        # Compute attacks used = initial_budget - min(left_attack)
        # Assuming initial budget is stored in args
        initial_budget = self.args.num_attack_train
        
        # Get minimum remaining budget per episode (final budget)
        final_budget = left_attack[:, -1]  # [batch_size, 1]
        attacks_used = initial_budget - final_budget  # [batch_size, 1]
        
        mean_attack_count = attacks_used.mean()
        
        return mean_attack_count

    # ========== Attacker Novelty Calculation ==========
    def calculate_attacker_behavioral_novelty(self, batch, attacker_idx, elite_attackers, population_attackers):
        """
        Calculate behavioral novelty of an attacker compared to elite archive.
        
        Behavioral Characterization:
        - Victim selection pattern: distribution of which agents are targeted
        - Attack timing: when attacks occur in the episode
        - Attack efficiency: coverage vs budget usage
        
        Args:
            batch: Episode batch data
            attacker_idx: Index of current attacker in population
            elite_attackers: Indices of elite attackers (archive)
            population_attackers: List of all attacker instances
            
        Returns:
            novelty_score: Average behavioral distance to elite archive
        """
        if len(elite_attackers) == 0:
            # No archive yet, maximum novelty
            return th.tensor(1.0, device=self.device)
        
        current_attacker = population_attackers[attacker_idx]
        current_attacker.eval()
        
        # Extract behavioral features for current attacker
        current_behavior = self._extract_attacker_behavior(batch, current_attacker)
        
        # Compute behavioral distance to each elite
        distances = []
        for elite_idx in elite_attackers:
            if elite_idx == attacker_idx:
                continue  # Skip self-comparison
            
            elite_attacker = population_attackers[elite_idx]
            elite_attacker.eval()
            
            elite_behavior = self._extract_attacker_behavior(batch, elite_attacker)
            
            # Compute behavioral distance (L2 norm in behavior space)
            distance = self._behavioral_distance(current_behavior, elite_behavior)
            distances.append(distance)
        
        if len(distances) == 0:
            return th.tensor(1.0, device=self.device)
        
        # Novelty = average distance to k-nearest neighbors (k=min(3, len(archive)))
        k = min(3, len(distances))
        distances_tensor = th.tensor(distances, device=self.device)
        topk_distances = th.topk(distances_tensor, k, largest=False).values
        novelty_score = th.mean(topk_distances)
        
        return novelty_score
    
    def _extract_attacker_behavior(self, batch, attacker):
        """
        Extract behavioral characterization of an attacker.
        
        Returns:
            behavior_vector: Dictionary containing:
                - victim_distribution: (n_agents+1,) distribution over victim choices
                - attack_timing: (time_bins,) distribution of when attacks occur
                - victim_entropy: Entropy of victim selection (diversity measure)
        """
        with th.no_grad():
            # Get victim selections across the batch
            victim_ids_list = []
            attack_q_values_list = []
            
            for t in range(batch.max_seq_length - 1):
                attacker_q = attacker.batch_forward(batch, t=t)  # (bs, n_agents+1)
                victim_id = th.argmax(attacker_q, dim=-1)  # (bs,) greedy selection
                
                victim_ids_list.append(victim_id)
                attack_q_values_list.append(attacker_q)
            
            victim_ids = th.stack(victim_ids_list, dim=1)  # (bs, seq_len-1)
            attack_q_values = th.stack(attack_q_values_list, dim=1)  # (bs, seq_len-1, n_agents+1)
            
            # Feature 1: Victim selection distribution
            victim_distribution = th.zeros(self.args.n_agents + 1, device=self.device)
            for i in range(self.args.n_agents + 1):
                victim_distribution[i] = (victim_ids == i).float().mean()
            
            # Feature 2: Attack timing distribution (early, mid, late episode)
            time_bins = 3
            seq_len = victim_ids.shape[1]
            attack_timing = th.zeros(time_bins, device=self.device)
            
            for bin_idx in range(time_bins):
                start_t = (seq_len * bin_idx) // time_bins
                end_t = (seq_len * (bin_idx + 1)) // time_bins
                
                # Count non-"no-attack" choices in this time bin
                bin_victims = victim_ids[:, start_t:end_t]
                attacks_in_bin = (bin_victims != self.args.n_agents).float().mean()
                attack_timing[bin_idx] = attacks_in_bin
            
            # Feature 3: Victim entropy (diversity of targeting)
            victim_probs = victim_distribution + 1e-8  # Add epsilon for stability
            victim_probs = victim_probs / victim_probs.sum()  # Normalize
            victim_entropy = -(victim_probs * th.log(victim_probs)).sum()
            
            # Combine into behavior vector
            behavior_vector = {
                'victim_distribution': victim_distribution,  # (n_agents+1,)
                'attack_timing': attack_timing,  # (time_bins,)
                'victim_entropy': victim_entropy.unsqueeze(0),  # (1,)
            }
            
            return behavior_vector
    
    def _behavioral_distance(self, behavior1, behavior2):
        """
        Compute distance between two behavioral characterizations.
        
        Uses weighted Euclidean distance across different behavior features.
        """
        # Weight different features
        w_victim = 1.0  # Victim selection pattern
        w_timing = 0.5  # Attack timing
        w_entropy = 0.3  # Victim diversity
        
        # Distance in victim selection space
        dist_victim = th.norm(behavior1['victim_distribution'] - behavior2['victim_distribution'], p=2)
        
        # Distance in attack timing space
        dist_timing = th.norm(behavior1['attack_timing'] - behavior2['attack_timing'], p=2)
        
        # Distance in entropy (scalar)
        dist_entropy = th.abs(behavior1['victim_entropy'] - behavior2['victim_entropy']).squeeze()
        
        # Weighted combination
        total_distance = (w_victim * dist_victim + 
                         w_timing * dist_timing + 
                         w_entropy * dist_entropy)
        
        return total_distance.item()
    
    # ========== Attacker Lamarckian SGD Finetuning ==========
    def lamarckian_finetune_attacker(self, attacker, episode_batch, num_sgd_steps=1):
        """
        Lamarckian SGD微调for Attacker: 沿着quality目标(negative defender reward)进行梯度下降。
        
        与defender的Lamarckian不同:
        - Defender优化: 最小化TD error (拟合环境)
        - Attacker优化: 最大化negative reward (破坏defender性能)
        
        训练目标:
        1. 主要: 最小化 defender reward (即最大化 -reward)
        2. 辅助: 保持Soft Q-Learning的策略稀疏性
        
        Args:
            attacker: MLPAttacker instance to finetune
            episode_batch: Recent episode batch for computing gradient
            num_sgd_steps: Number of SGD steps to perform
            
        Returns:
            final_loss: Final quality loss value (for monitoring)
        """
        # 1. 设置为训练模式
        attacker.train()
        
        # 2. 创建临时优化器 (使用较小学习率，避免破坏进化得到的权重)
        finetune_lr = getattr(self.args, 'attacker_finetune_lr', self.args.attack_lr * 0.1)
        finetune_optimizer = th.optim.RMSprop(
            attacker.parameters(), 
            lr=finetune_lr,
            alpha=self.args.optim_alpha, 
            eps=self.args.optim_eps
        )
        
        # 准备batch
        max_ep_t = episode_batch.max_t_filled()
        batch = episode_batch[:, :max_ep_t]
        
        if batch.device != self.args.device:
            batch.to(self.args.device)
        
        # 提取数据
        rewards = batch["reward"][:, :-1]  # Defender's reward
        # if self.args.shaping_reward:
        #     rewards = batch["shaping_reward"][:, :-1]
        
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # 3. 执行多步SGD微调
        for step in range(num_sgd_steps):
            # Forward pass: 计算attacker的Q值
            attacker_qs = []
            for t in range(batch.max_seq_length):
                attacker_q = attacker.batch_forward(batch, t=t)
                attacker_qs.append(attacker_q)
            attacker_qs = th.stack(attacker_qs, dim=1)  # (bs, seq_len, n_agents+1)
            
            # === Quality Objective: Maximize negative defender reward ===
            # 策略: 选择能最大化负reward的victim
            # 使用softmax得到victim选择概率分布
            victim_logits = attacker_qs[:, :-1]  # (bs, seq_len-1, n_agents+1)
            victim_probs = th.softmax(victim_logits, dim=-1)  # Temperature=0.1 for sharper distribution
            
            # 计算期望的victim选择下的reward
            # 假设: 选择不同victim会导致不同的reward
            # 简化为: 最大化选择"攻击"行为(非no-attack)的概率
            attack_prob = 1.0 - victim_probs[:, :, -1]  # Probability of NOT choosing "no-attack"
            
            # Quality loss: 最小化defender reward，加权于攻击概率
            # L_quality = E[reward * attack_prob] = 期望的defender损失
            quality_loss = -(rewards.squeeze(-1) * attack_prob * mask.squeeze(-1)).sum() / mask.sum()
            
            # === Regularization: Soft Q-Learning稀疏性约束 ===
            # 保持与p_ref的一致性，防止过度偏离稀疏攻击策略
            # KL散度: D_KL(victim_probs || p_ref)
            p_ref_expanded = attacker.p_ref.unsqueeze(0).unsqueeze(0).expand_as(victim_probs)
            kl_div = (victim_probs * (th.log(victim_probs + 1e-8) - th.log(p_ref_expanded + 1e-8))).sum(dim=-1)
            
            reg_weight = getattr(self.args, 'attacker_finetune_reg', 0.01)
            reg_loss = (kl_div * mask.squeeze(-1)).sum() / (mask.sum() + 1e-8)
            
            # Total loss
            total_loss = quality_loss + reg_weight * reg_loss
            
            # 检查loss是否合理
            if not th.isfinite(total_loss):
                # Loss异常，停止微调
                return float('inf')
            
            # Backward and optimize
            finetune_optimizer.zero_grad()
            total_loss.backward()
            
            # 梯度裁剪 (更保守)
            grad_clip = getattr(self.args, 'attacker_finetune_grad_clip', self.args.grad_norm_clip * 0.5)
            grad_norm = th.nn.utils.clip_grad_norm_(attacker.parameters(), grad_clip)
            
            # 检查梯度
            if not th.isfinite(grad_norm):
                finetune_optimizer.zero_grad()
                return float('inf')
            
            finetune_optimizer.step()
        
        # 4. 设置回eval模式
        attacker.eval()
        
        # 5. 返回最终loss (主要是quality loss)
        return quality_loss.item()

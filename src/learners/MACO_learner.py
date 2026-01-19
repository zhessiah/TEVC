import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.enriched_qmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
import torch.nn.functional as F
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from utils.attack_util import logits_margin, attack, get_diff, noise_atk, get_max_diff
# from torch.autograd import Variable
# from utils.pareto import MinNormSolver, gradient_normalizers
from controllers import REGISTRY as mac_REGISTRY


class MACOLearner:
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


    # ========== SIMPLIFIED DEFENDER FITNESS FUNCTIONS (2 objectives only) ==========
    # Following MACO refactoring:
    # Defender Fitness (2 objectives):
    # 1. Optimality: TD Error (calculate_TD_error) - environment fitting accuracy
    # 2. Robustness: Fault Isolation Ratio (calculate_adversarial_loss) - Byzantine fault tolerance
    #
    # REMOVED deprecated functions (moved to implicit mechanisms):
    # - calculate_confidence_Q: Redundant with TD error
    # - calculate_adversarial_entropy: Complexity reduction  
    # - calculate_influence_constraint: Complexity reduction
    # - calculate_evolutionary_consensus: Implicit via Pareto front
    # - calculate_adversarial_novelty: Diversity via exploration
    # =================================================================================
    
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
    
    # ========== REMOVED DEPRECATED FITNESS FUNCTIONS ==========
    # Following MACO refactoring - complexity reduction
    # Removed functions:
    # - calculate_influence_constraint (权力制衡约束)
    # - calculate_evolutionary_consensus (种群共识校准)  
    # - calculate_adversarial_novelty (新颖性搜索)
    # - _js_divergence (helper function)
    # - update_elite_archive (helper function)
    #
    # Rationale: Simplified to 2-objective optimization for faster convergence
    # Diversity is maintained through exploration (epsilon-greedy)
    # ===========================================================
    
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
    
#     def memetic_finetune(self, pop_genome, batch, num_sgd_steps=1):
#         """
#         DEPRECATED: This function is kept for backward compatibility only.
#         According to MACO refactoring, explicit memetic learning is REMOVED.
#         
#         Rationale from document:
#         "我们可以novel一些，不再说'memetic learning' 而是'memetic injection模因注入/文化注入'。
#         然后去掉原本的memetic learning的过程。"
#         
#         The main agent learning (train()) itself IS the implicit memetic mechanism:
#         - Main agent learns via SGD (TD error minimization)
#         - Parameters are injected to replace dominated individuals (memetic injection)
#         - This is equivalent to Value-Evolutionary-Based Reinforcement Learning
#         
#         DO NOT CALL THIS FUNCTION IN NEW CODE.
#         Use only: rl_to_evo_excluding_elites() for parameter injection.
#         
#         Original description:
#         Memetic SGD Injection: 对种群中的单个 Genome 进行 SGD 微调。
#         
#         **Memetic 特性**: 微调直接修改个体的权重（基因），因此获得的改进会遗传给后代。
#         与达尔文进化不同，这里"后天学习"的结果会改变遗传信息。
#         
#         Args:
#             pop_genome: 要微调的 Genome（来自 self.pop_genome）
#             batch: EpisodeBatch 用于 SGD 训练
#             num_sgd_steps: SGD 迭代次数（默认 1）
#             
#         Returns:
#             td_loss: 微调后的 TD 损失（用于监控）
#         """
#         # 1. 为该个体创建临时优化器（避免影响主 RL 优化器）
#         genome_params = list(pop_genome.parameters())
#         
#         # 使用更保守的学习率进行微调（防止破坏进化得到的结构和梯度爆炸）
#         finetune_lr = getattr(self.args, 'finetune_lr', self.args.lr * 0.01)  # 改为 0.01 (更保守)
#         
#         if self.args.optimizer == 'adam':
#             genome_optimizer = Adam(params=genome_params, lr=finetune_lr)
#         else:
#             genome_optimizer = RMSprop(params=genome_params, lr=finetune_lr, 
#                                       alpha=self.args.optim_alpha, eps=self.args.optim_eps)
#         
#         # 初始化 loss（防止循环内跳过导致未定义）
#         L_td = th.tensor(0.0, device=self.device)
#         
#         # 2. 执行 SGD 步骤
#         for _ in range(num_sgd_steps):
#             # Get the relevant quantities
#             rewards = batch["reward"][:, :-1]
#             actions = batch["actions"][:, :-1]
#             terminated = batch["terminated"][:, :-1].float()
#             mask = batch["filled"][:, :-1].float()
#             mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
#             avail_actions = batch["avail_actions"]
#             
#             # Set to train mode
#             pop_genome.train()
#             
#             # Forward pass for mac1
#             mac_out_1 = []
#             pop_genome.mac1.init_hidden(batch.batch_size)
#             for t in range(batch.max_seq_length):
#                 agent_outs_1 = pop_genome.mac1.forward(batch, t=t)
#                 mac_out_1.append(agent_outs_1)
#             
#             # Forward pass for mac2
#             mac_out_2 = []
#             pop_genome.mac2.init_hidden(batch.batch_size)
#             for t in range(batch.max_seq_length):
#                 agent_outs_2 = pop_genome.mac2.forward(batch, t=t)
#                 mac_out_2.append(agent_outs_2)
#             
#             mac_out_1 = th.stack(mac_out_1, dim=1)
#             mac_out_2 = th.stack(mac_out_2, dim=1)
#             
#             # Pick Q-values for actions taken
#             chosen_action_qvals_1 = th.gather(mac_out_1[:, :-1], dim=3, index=actions).squeeze(3)
#             chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim=3, index=actions).squeeze(3)
#             
#             # Calculate targets using pop_genome itself (detached to avoid gradients)
#             # 关键修复：不使用 self.target_Genome，而是用 pop_genome 自己的 detached 版本
#             with th.no_grad():
#                 # 使用 pop_genome 自己来计算目标（而非主 RL 的 target 网络）
#                 pop_genome.eval()  # 临时设为 eval 模式计算目标
#                 
#                 # 计算目标 Q 值
#                 target_mac_out_1 = []
#                 pop_genome.mac1.init_hidden(batch.batch_size)
#                 for t in range(batch.max_seq_length):
#                     target_agent_outs = pop_genome.mac1.forward(batch, t=t)
#                     target_mac_out_1.append(target_agent_outs)
#                 
#                 target_mac_out_2 = []
#                 pop_genome.mac2.init_hidden(batch.batch_size)
#                 for t in range(batch.max_seq_length):
#                     target_agent_outs = pop_genome.mac2.forward(batch, t=t)
#                     target_mac_out_2.append(target_agent_outs)
#                 
#                 target_mac_out_1 = th.stack(target_mac_out_1, dim=1)
#                 target_mac_out_2 = th.stack(target_mac_out_2, dim=1)
#                 
#                 # 使用 min(Q1, Q2) 作为目标（Double Q-learning 的保守估计）
#                 target_mac_out = th.min(target_mac_out_1, target_mac_out_2)
#                 
#                 # Double Q-learning: 使用当前网络选择动作，目标网络评估
#                 mac_out_combined = th.min(mac_out_1, mac_out_2)
#                 mac_out_detach = mac_out_combined.clone().detach()
#                 mac_out_detach[avail_actions == 0] = -9999999
#                 cur_max_actions = mac_out_detach.max(dim=3, keepdim=True)[1]
#                 target_max_qvals = th.gather(target_mac_out, 3, cur_max_actions).squeeze(3)
#                 
#                 # 使用 pop_genome 的 mixer（而非 self.target_mixer）
#                 target_max_qvals = self.mixer(target_max_qvals, batch["state"])
#                 
#                 if getattr(self.args, 'q_lambda', False):
#                     qvals = th.gather(target_mac_out, 3, batch["actions"]).squeeze(3)
#                     qvals = self.mixer(qvals, batch["state"])
#                     targets = build_q_lambda_targets(rewards, terminated, mask, target_max_qvals, qvals,
#                                                     self.args.gamma, self.args.td_lambda)
#                 else:
#                     targets = build_td_lambda_targets(rewards, terminated, mask, target_max_qvals,
#                                                      self.args.n_agents, self.args.gamma, self.args.td_lambda)
#                 
#                 pop_genome.train()  # 恢复训练模式
#             
#             # Apply mixer from pop_genome
#             chosen_action_qvals_1 = self.mixer(chosen_action_qvals_1, batch["state"][:, :-1])
#             chosen_action_qvals_2 = self.mixer(chosen_action_qvals_2, batch["state"][:, :-1])
#             
#             # Calculate TD error
#             td_error_1 = (chosen_action_qvals_1 - targets.detach())
#             td_error2_1 = 0.5 * td_error_1.pow(2)
#             
#             td_error_2 = (chosen_action_qvals_2 - targets.detach())
#             td_error2_2 = 0.5 * td_error_2.pow(2)
#             
#             mask_expanded = mask.expand_as(td_error2_1)
#             masked_td_error_1 = td_error2_1 * mask_expanded
#             masked_td_error_2 = td_error2_2 * mask_expanded
#             
#             # Total loss
#             L_td_1 = masked_td_error_1.sum() / (mask_expanded.sum() + 1e-8)
#             L_td_2 = masked_td_error_2.sum() / (mask_expanded.sum() + 1e-8)
#             L_td = L_td_1 + L_td_2
#             
#             # 检查 loss 是否有效（防止 NaN/Inf 传播）
#             if not th.isfinite(L_td):
#                 # Loss 无效，跳过此次更新
#                 return float('inf')
#             
#             # 限制 loss 的最大值（防止梯度爆炸）
#             max_loss = getattr(self.args, 'max_finetune_loss', 100.0)
#             if L_td.item() > max_loss:
#                 # Loss 过大，使用裁剪后的 loss
#                 L_td = th.clamp(L_td, max=max_loss)
#             
#             # Optimize
#             genome_optimizer.zero_grad()
#             L_td.backward()
#             
#             # 更严格的梯度裁剪
#             grad_norm_clip = getattr(self.args, 'finetune_grad_clip', self.args.grad_norm_clip * 0.5)
#             grad_norm = th.nn.utils.clip_grad_norm_(genome_params, grad_norm_clip)
#             
#             # 检查梯度是否有效
#             if not th.isfinite(grad_norm):
#                 # 梯度无效，跳过此次更新
#                 genome_optimizer.zero_grad()
#                 return float('inf')
#             
#             genome_optimizer.step()
#         
#         # 3. 返回最终损失（用于监控微调效果）
#         return L_td.item()
# 
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
#     def calculate_attacker_quality(self, batch, attacker_idx, population_attackers):
#         """
#         DEPRECATED: This function is no longer used in run.py (refactoring).
#         
#         The correct approach is to run environment with EACH attacker separately in run.py:
#         ```python
#         genome.mac1.set_attacker(population_attackers[i])
#         attacker_eval_batch = runner.run(genome, test_mode=False)
#         quality = -attacker_eval_batch["reward"].mean()
#         ```
#         
#         Kept for backward compatibility only.
#         
#         Original (incorrect) implementation:
#         Calculate attack quality: negative mean episode return (lower defender reward = better).
#         
#         Problem: This function computed quality from a shared batch, causing all attackers
#         to have identical quality values. Fixed in by running separate evaluations.
#         
#         Args:
#             batch: Episode batch data (SHARED - this was the bug!)
#             attacker_idx: Index of attacker in population (unused - this was the bug!)
#             population_attackers: List of all attacker instances
#             
#         Returns:
#             mean_return: Mean episode return (scalar tensor)
#         """
#         # Extract rewards from batch
#         rewards = batch["reward"][:, :-1]  # [batch_size, T, 1]
#         mask = batch["filled"][:, :-1].float()  # [batch_size, T, 1]
#         
#         # Compute episode returns
#         episode_returns = (rewards * mask).sum(dim=1) / mask.sum(dim=1)  # [batch_size, 1]
#         mean_return = episode_returns.mean()
#         
#         return mean_return
#     
    def calculate_attacker_TD_error(self, batch: EpisodeBatch, attacker_idx: int, population_attackers):
        """
        CORRECTED: Calculate attacker fitness via TD Error maximization with COUNTERFACTUAL ACTIONS.
        
        Mathematical Foundation:
        F_att,1(φ) = E_τ~D_φ [(y - Q_tot(s, a_φ))²]
        
        Where:
        - y: TD target (computed from clean target network)
        - a_φ: Counterfactual actions under attacker φ's Byzantine interference
        - Attacker φ selects victims and replaces their actions with Byzantine (worst) actions
        - Different attackers → different victim selections → different action sequences → different Q-values
        
        KEY INSIGHT (Byzantine Attack Model):
        - Attacker does NOT perturb observations (ũ_φ)
        - Attacker perturbs ACTIONS via victim selection
        - We use batch's existing `byzantine_actions` but apply attacker-specific victim masks
        
        CRITICAL FIX: 
        - Previous version used batch["actions"] which were collected WITHOUT this attacker
        - Now we construct counterfactual actions: apply THIS attacker's victim selection to batch
        - This makes different attackers produce DIFFERENT TD errors (different victims → different actions)
        
        Computational Efficiency:
        1. No environment rollout needed (offline batch computation)
        2. Reuse batch's `byzantine_actions` (pre-computed worst actions)
        3. Only recompute victim selection (cheap: attacker.batch_forward())
        
        Args:
            batch: EpisodeBatch from replay buffer
                   Must contain: actions, byzantine_actions, obs, state, reward, terminated
            attacker_idx: Index of attacker to evaluate
            population_attackers: List of all attacker instances
            
        Returns:
            td_error: Mean TD error under this attacker's Byzantine interference
                     Higher value = better attacker (more disruptive)
        """
        # Get the attacker to evaluate
        attacker = population_attackers[attacker_idx]
        attacker.eval()
        
        # === Step 0: Construct counterfactual actions using THIS attacker's victim selection ===
        # Byzantine attack: attacker selects which agents to attack, then replaces their actions
        batch_size = batch.batch_size
        seq_len = batch.max_seq_length
        n_agents = self.args.n_agents
        
        # Get base actions and Byzantine actions from batch
        original_actions = batch["actions"]  # [bs, seq, n_agents, 1]
        byzantine_actions = batch["byzantine_actions"]  # [bs, seq, n_agents, 1]
        
        # Use THIS attacker to select victims for each timestep
        # attacker.batch_forward() returns logits over (n_agents + 1) actions
        # where index i means "attack agent i", index n_agents means "no attack"
        counterfactual_actions = original_actions.clone()  # Start with original actions
        
        with th.no_grad():
            for t in range(seq_len):
                # Get attacker's victim selection at timestep t
                attacker_logits = attacker.batch_forward(batch, t)  # [bs, n_agents+1]
                victim_id = attacker_logits.argmax(dim=1)  # [bs] - which agent to attack
                
                # Apply Byzantine attack: replace victim's action with Byzantine action
                for b in range(batch_size):
                    victim = victim_id[b].item()
                    if victim < n_agents:  # Valid agent (not "no attack")
                        # Replace this agent's action with Byzantine action
                        counterfactual_actions[b, t, victim] = byzantine_actions[b, t, victim]
        
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1]
        actions_counterfactual = counterfactual_actions[:, :-1]  # Use counterfactual actions!
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Set to eval mode for evaluation (no dropout, no exploration)
        self.Genome.eval()
        
        with th.no_grad():
            # === Step 1: Forward pass to get Q-values ===
            # NOTE: We DON'T set attacker on MACs because Byzantine attack affects ACTIONS, not observations
            # The disruption comes from counterfactual_actions, not from perturbed observations
            
            # Calculate Q-Values using mac1
            mac_out_1 = []
            self.Genome.mac1.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.Genome.mac1.forward(batch, t=t)
                mac_out_1.append(agent_outs)
            mac_out_1 = th.stack(mac_out_1, dim=1)
            
            # Calculate Q-Values using mac2
            mac_out_2 = []
            self.Genome.mac2.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.Genome.mac2.forward(batch, t=t)
                mac_out_2.append(agent_outs)
            mac_out_2 = th.stack(mac_out_2, dim=1)
        
        # Pick Q-Values for the COUNTERFACTUAL actions (attacker-perturbed actions)
        # This is the key: different attackers → different victim selections → different counterfactual actions
        chosen_action_qvals_1 = th.gather(mac_out_1[:, :-1], dim=3, index=actions_counterfactual).squeeze(3)
        chosen_action_qvals_2 = th.gather(mac_out_2[:, :-1], dim=3, index=actions_counterfactual).squeeze(3)
        
        # === Step 2: Calculate TD targets (using target network with clean actions) ===
        # Target network uses ORIGINAL actions (not counterfactual)
        # This measures how much attacker's action perturbation disrupts Q-value accuracy
        with th.no_grad():
            self.target_Genome.eval()
            
            # Target Q-values from target_Genome
            target_mac_out = []
            self.target_Genome.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_Genome.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            target_mac_out = th.stack(target_mac_out, dim=1)

            # Double Q-learning: use current network for action selection
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

        # === Step 3: Apply mixer and compute TD error ===
        with th.no_grad():
            chosen_action_qvals_1 = self.mixer(chosen_action_qvals_1, batch["state"][:, :-1])
            chosen_action_qvals_2 = self.mixer(chosen_action_qvals_2, batch["state"][:, :-1])

        # Calculate TD error for both MACs
        td_error_1 = (chosen_action_qvals_1 - targets.detach())
        td_error2_1 = 0.5 * td_error_1.pow(2)
        
        td_error_2 = (chosen_action_qvals_2 - targets.detach())
        td_error2_2 = 0.5 * td_error_2.pow(2)

        mask_expanded = mask.expand_as(td_error2_1)
        masked_td_error_1 = td_error2_1 * mask_expanded
        masked_td_error_2 = td_error2_2 * mask_expanded

        # Average TD errors from both MACs
        mean_td_error_1 = masked_td_error_1.sum() / (mask_expanded.sum() + 1e-8)
        mean_td_error_2 = masked_td_error_2.sum() / (mask_expanded.sum() + 1e-8)
        mean_td_error = (mean_td_error_1 + mean_td_error_2) / 2.0
        
        # Safety check: if TD error is unreasonably large, cap it
        if mean_td_error.item() > 1e6:
            return th.tensor(1e6, device=mean_td_error.device)
        
        # Return TD error (higher = better for attacker, as it means more disruption)
        return mean_td_error
    
    # ========== REMOVED: calculate_attacker_efficiency ==========
    # "原来的3个目标去除掉efficiency，因为实际会导致完全不攻击。
    # （一点预算都不用，就是最高效的，不会被pareto支配）"
    #
    # Efficiency leads to no attacks at all (zero budget = maximum efficiency),
    # which defeats the purpose of adversarial training.
    #
    # Attacker Fitness (2 objectives only):
    # 1. Quality: Negative defender reward (calculate_attacker_quality)
    # 2. Novelty: Attack pattern diversity (calculate_attacker_behavioral_novelty)
    # =================================================================

    # ========== Attacker Novelty Calculation (REFACTORED: Batch KL-Divergence) ==========
    def calculate_all_attacker_novelties(self, batch, population_attackers):
        """
        OPTIMIZED: Calculate novelty for ALL attackers in ONE pass (eliminates N² redundancy).
        
        Previous approach: Called calculate_attacker_behavioral_novelty() N times
        - Each call recomputed all N distributions → O(N²) complexity
        - Wasteful: Same distributions computed N times
        
        NEW APPROACH (inspired by population.py d_loss):
        - Compute action probability distribution π_i(a|s,k) for each attacker ONCE
        - Uses ALL timesteps in the episode (complete attack pattern evaluation)
        - Calculate population consensus: π_mean(a|s,k) = (1/N) Σ π_i(a|s,k)
        - Novelty_i = KL(π_i || π_mean) for all i simultaneously
        
        Key Innovation:
        - O(N) complexity instead of O(N²)
        - Evaluates complete attack behavior over entire episode
        - Higher KL divergence → more different from population → higher novelty
        - Direct policy-space diversity (vs. behavior-space distance)
        
        Args:
            batch: Episode batch data
            population_attackers: List of all attacker instances
            
        Returns:
            novelty_scores: List of novelty scores (one per attacker), length = pop_size
        """
        pop_size = len(population_attackers)
        
        # Edge case: single attacker has no diversity reference
        if pop_size == 1:
            return [0.0]
        
        # Step 1: Get action distributions for all attackers over ALL timesteps in episode
        # This captures the complete attack pattern for the entire episode
        all_action_dists = []
        for attacker in population_attackers:
            attacker.eval()
            # Get distribution for each timestep in the episode
            action_dist = self._get_attacker_action_distribution_batch(attacker, batch)
            all_action_dists.append(action_dist)
        
        # Step 2: Calculate population consensus (mean distribution)
        all_action_dists = th.stack(all_action_dists, dim=0)  # (pop_size, bs*seq_len, n_agents+1)
        mean_action_dist = all_action_dists.mean(dim=0)  # (bs*seq_len, n_agents+1)
        
        # Step 3: Calculate KL divergence for ALL attackers simultaneously
        novelty_scores = []
        for i in range(pop_size):
            current_dist = all_action_dists[i]  # (bs*seq_len, n_agents+1)
            
            # KL(current || mean) = Σ p(x) log(p(x)/q(x))
            # Using F.kl_div: expects log(q) as input, p as target
            kl_divergence = F.kl_div(
                th.log(mean_action_dist),  # log of mean (reference)
                current_dist,  # current distribution (target)
                reduction='batchmean'
            )
            
            novelty_scores.append(kl_divergence.item())
        
        return novelty_scores
    
#     def calculate_attacker_behavioral_novelty(self, batch, attacker_idx, elite_attackers, population_attackers):
#         """
#         DEPRECATED: Use calculate_all_attacker_novelties() instead for O(N) efficiency.
#         
#         This function is kept for backward compatibility only.
#         It internally calls the optimized batch version.
#         
#         Args:
#             batch: Episode batch data
#             attacker_idx: Index of current attacker in population
#             elite_attackers: Indices of elite attackers (archive) [UNUSED]
#             population_attackers: List of all attacker instances
#             
#         Returns:
#             novelty_score: KL divergence from population consensus
#         """
#         # Call the optimized batch version and extract the specific attacker's novelty
#         all_novelties = self.calculate_all_attacker_novelties(batch, population_attackers)
#         return th.tensor(all_novelties[attacker_idx], device=self.device)
#     
    def _get_attacker_action_distribution_batch(self, attacker, batch):
        """
        Get action probability distribution for an attacker over ALL timesteps in episode batch.
        
        Uses attacker.batch_forward() to compute distributions for entire episode.
        This captures the complete attack pattern: π(a|s,k) for all (s,k) in episode.
        
        Policy: π(a|s,k) = softmax(logits(s,k))
        - Attacker network directly outputs logits over victim choices (n_agents+1 actions)
        - Apply softmax to convert to probability distribution
        
        Args:
            attacker: Attacker network
            batch: Episode batch data
            
        Returns:
            action_probs: (bs*seq_len, n_agents+1) probability distribution over all timesteps
        """
        with th.no_grad():
            bs = batch.batch_size
            seq_len = batch.max_seq_length
            
            # Collect Q-values for all timesteps
            all_q_values = []
            for t in range(seq_len):
                attacker_logits = attacker.batch_forward(batch, t)  # (bs, n_agents+1)
                all_q_values.append(attacker_logits)
            
            all_q_values = th.stack(all_q_values, dim=1)  # (bs, seq_len, n_agents+1)
            
            # Reshape to (bs*seq_len, n_agents+1) for batch processing
            logits_flat = all_q_values.reshape(-1, self.args.n_agents + 1)
            
            # Directly apply softmax to attacker's logits
            # The attacker network outputs logits, not Q-values
            # π(a|s,k) = softmax(logits(s,k))
            action_probs = F.softmax(logits_flat, dim=-1)  # (bs*seq_len, n_agents+1)
            
            return action_probs
    
    # ========== OLD BEHAVIOR-BASED NOVELTY FUNCTIONS (DEPRECATED) ==========
    # Kept for reference, but no longer used in new KL-divergence approach
    
#     def _extract_attacker_behavior(self, batch, attacker):
#         """
#         [DEPRECATED] Extract behavioral characterization of an attacker.
#         
#         Returns:
#             behavior_vector: Dictionary containing:
#                 - victim_distribution: (n_agents+1,) distribution over victim choices
#                 - attack_timing: (time_bins,) distribution of when attacks occur
#                 - victim_entropy: Entropy of victim selection (diversity measure)
#         """
#         with th.no_grad():
#             # Get victim selections across the batch
#             victim_ids_list = []
#             attack_q_values_list = []
#             
#             for t in range(batch.max_seq_length - 1):
#                 attacker_q = attacker.batch_forward(batch, t=t)  # (bs, n_agents+1)
#                 victim_id = th.argmax(attacker_q, dim=-1)  # (bs,) greedy selection
#                 
#                 victim_ids_list.append(victim_id)
#                 attack_q_values_list.append(attacker_q)
#             
#             victim_ids = th.stack(victim_ids_list, dim=1)  # (bs, seq_len-1)
#             attack_q_values = th.stack(attack_q_values_list, dim=1)  # (bs, seq_len-1, n_agents+1)
#             
#             # Feature 1: Victim selection distribution
#             victim_distribution = th.zeros(self.args.n_agents + 1, device=self.device)
#             for i in range(self.args.n_agents + 1):
#                 victim_distribution[i] = (victim_ids == i).float().mean()
#             
#             # Feature 2: Attack timing distribution (early, mid, late episode)
#             time_bins = 3
#             seq_len = victim_ids.shape[1]
#             attack_timing = th.zeros(time_bins, device=self.device)
#             
#             for bin_idx in range(time_bins):
#                 start_t = (seq_len * bin_idx) // time_bins
#                 end_t = (seq_len * (bin_idx + 1)) // time_bins
#                 
#                 # Count non-"no-attack" choices in this time bin
#                 bin_victims = victim_ids[:, start_t:end_t]
#                 attacks_in_bin = (bin_victims != self.args.n_agents).float().mean()
#                 attack_timing[bin_idx] = attacks_in_bin
#             
#             # Feature 3: Victim entropy (diversity of targeting)
#             victim_probs = victim_distribution + 1e-8  # Add epsilon for stability
#             victim_probs = victim_probs / victim_probs.sum()  # Normalize
#             victim_entropy = -(victim_probs * th.log(victim_probs)).sum()
#             
#             # Combine into behavior vector
#             behavior_vector = {
#                 'victim_distribution': victim_distribution,  # (n_agents+1,)
#                 'attack_timing': attack_timing,  # (time_bins,)
#                 'victim_entropy': victim_entropy.unsqueeze(0),  # (1,)
#             }
#             
#             return behavior_vector
#     
#     def _behavioral_distance(self, behavior1, behavior2):
#         """
#         Compute distance between two behavioral characterizations.
#         
#         Uses weighted Euclidean distance across different behavior features.
#         """
#         # Weight different features
#         w_victim = 1.0  # Victim selection pattern
#         w_timing = 0.5  # Attack timing
#         w_entropy = 0.3  # Victim diversity
#         
#         # Distance in victim selection space
#         dist_victim = th.norm(behavior1['victim_distribution'] - behavior2['victim_distribution'], p=2)
#         
#         # Distance in attack timing space
#         dist_timing = th.norm(behavior1['attack_timing'] - behavior2['attack_timing'], p=2)
#         
#         # Distance in entropy (scalar)
#         dist_entropy = th.abs(behavior1['victim_entropy'] - behavior2['victim_entropy']).squeeze()
#         
#         # Weighted combination
#         total_distance = (w_victim * dist_victim + 
#                          w_timing * dist_timing + 
#                          w_entropy * dist_entropy)
#         
#         return total_distance.item()
#     
#     # ========== Attacker Memetic SGD Training (NEW APPROACH) ==========
    def memetic_finetune_attacker(self, attacker, episode_batch, num_sgd_steps=1):
        """
        REFACTORED: Memetic SGD for Attacker using Budget-Modulated Attack Advantage.
        Replaces old Soft Q-Learning with policy gradient-like approach.
        
        New Training Mechanism (MACO):
        1. Attack Advantage Function: A_att(j|s,k) = Q_j(s,u_j) - γ(k)·Q̄(s)
        2. Budget Modulation: γ(k) = 1 + budget_lambda(K-k)/K (higher threshold when budget is scarce)
        3. Policy Gradient Loss: L_att = -E[Σ A_att(j)·log π(j)]
        
        Key Innovation:
        - Teaches attacker "opportunity cost awareness" (好钢用在刀刃上)
        - When budget is low, only attack high-value targets
        - When budget is high, attack above-average targets
        
        Args:
            attacker: MLPAttacker instance to finetune
            episode_batch: Recent episode batch for computing gradient
            num_sgd_steps: Number of SGD steps to perform
            
        Returns:
            final_loss: Final loss value (for monitoring)
        """
        # 1. Set to train mode
        attacker.train()
        
        # 2. Create temporary optimizer (conservative learning rate)
        finetune_lr = getattr(self.args, 'attacker_finetune_lr', self.args.attack_lr * 0.1)
        finetune_optimizer = th.optim.RMSprop(
            attacker.parameters(), 
            lr=finetune_lr,
            alpha=self.args.optim_alpha, 
            eps=self.args.optim_eps
        )
        
        # Prepare batch
        max_ep_t = episode_batch.max_t_filled()
        batch = episode_batch[:, :max_ep_t]
        
        if batch.device != self.args.device:
            batch.to(self.args.device)
        
        # Extract data
        rewards = batch["reward"][:, :-1]  # Defender's reward
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        left_attack = batch["left_attack"][:, :-1]  # Remaining budget k
        
        # Get attack budget K
        K = getattr(self.args, 'attack_num', 10)
        budget_lambda = getattr(self.args, 'budget_lambda', 1.0)  # Budget modulation strength
        
        # 3. Multi-step SGD finetuning
        for step in range(num_sgd_steps):
            # === Step 1: Get defender's local Q-values for all agents ===
            # We need Q_j(s, u_j) for each agent j to compute attack advantage
            # Use the main RL genome to get Q-values
            self.Genome.eval()
            self.Genome.init_hidden(batch.batch_size)
            
            agent_q_list = []
            with th.no_grad():
                for t in range(batch.max_seq_length):
                    agent_q = self.Genome.forward(batch, t=t)  # (bs, n_agents, n_actions)
                    agent_q_list.append(agent_q)
            
            agent_q = th.stack(agent_q_list, dim=1)[:, :-1]  # (bs, seq_len-1, n_agents, n_actions)
            
            # Get Q-values for executed actions
            actions = batch["actions"][:, :-1]  # (bs, seq_len-1, n_agents, 1)
            chosen_q = th.gather(agent_q, dim=3, index=actions).squeeze(3)  # (bs, seq_len-1, n_agents)
            
            # === Step 2: Compute Attack Advantage ===
            # Average Q across all agents: Q̄(s) = (1/N) Σ Q_j(s, u_j)
            Q_bar = chosen_q.mean(dim=2, keepdim=True)  # (bs, seq_len-1, 1)
            
            # Budget modulation factor: γ(k) = 1 + budget_lambda(K-k)/K
            gamma_k = 1.0 + budget_lambda * (K - left_attack) / K  # (bs, seq_len-1, 1)
            
            # Attack Advantage for each agent j: A_att(j) = Q_j - γ(k)·Q̄
            attack_advantage = chosen_q - gamma_k * Q_bar  # (bs, seq_len-1, n_agents)
            
            # For "no-attack" option (j=N+1), advantage is 0
            no_attack_advantage = th.zeros_like(attack_advantage[:, :, :1])  # (bs, seq_len-1, 1)
            attack_advantage_full = th.cat([attack_advantage, no_attack_advantage], dim=2)  # (bs, seq_len-1, n_agents+1)
            
            # === Step 3: Get attacker's action probabilities ===
            attacker_logits = []
            for t in range(batch.max_seq_length):
                logit = attacker.batch_forward(batch, t=t)
                attacker_logits.append(logit)
            attacker_logits = th.stack(attacker_logits, dim=1)[:, :-1]  # (bs, seq_len-1, n_agents+1)
            
            # Convert to log probabilities
            log_probs = th.log_softmax(attacker_logits, dim=-1)  # (bs, seq_len-1, n_agents+1)
            
            # === Step 4: Policy Gradient Loss ===
            # L_att = -E[Σ_j A_att(j) · log π(j)]
            # If A_att(j) > 0: High advantage → increase log π(j)  
            # If A_att(j) < 0: Low advantage → decrease log π(j)
            policy_loss = -(attack_advantage_full * log_probs * mask).sum() / mask.sum()
            
            # Check if loss is valid
            if not th.isfinite(policy_loss):
                return float('inf')
            
            # Backward and optimize
            finetune_optimizer.zero_grad()
            policy_loss.backward()
            grad_norm = th.nn.utils.clip_grad_norm_(attacker.parameters(), self.args.grad_norm_clip)
            finetune_optimizer.step()
        
        # 4. Return final loss for monitoring
        return policy_loss.item()
    
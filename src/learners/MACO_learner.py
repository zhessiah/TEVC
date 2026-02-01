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
        self.Genome = genome  # Single Genome (single MAC) for RL training
        self.logger = logger
        self.scheme = scheme  # Save scheme for creating population
        
        self.last_target_update_episode = 0
        self.device = th.device('cuda' if args.use_cuda  else 'cpu')
        self.params = list(genome.parameters())  # This includes MAC parameters

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
        # Each Genome contains: single MAC only 
        self.pop_size = args.pop_size
        self.elite_size = args.elite_size
        
        self.pop_genome = pop_genome  # Population of Genomes 
        
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
        # Get the Genome from population (contains single mac)
        pop_genome = self.pop_genome[genome_index]
        
        # Get the relevant quantities (same as in train())
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Set agent to eval mode for evaluation
        pop_genome.mac.agent.eval()
        
        with th.no_grad():  # No gradients needed for evaluation
            # Calculate Q-Values using mac
            mac_out = []
            pop_genome.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = pop_genome.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)
        
        # Pick the Q-Values for the actions taken
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        
        # Calculate the Q-Values necessary for the target 
        with th.no_grad():
            self.target_Genome.mac.agent.eval()
            
            # Target Q-values from target_Genome
            target_mac_out = []
            self.target_Genome.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_Genome.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            target_mac_out = th.stack(target_mac_out, dim=1)

            # Max over target Q-Values / Double Q learning 
            mac_out_detach = mac_out.clone().detach()
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

        # Mixer
        # Use the mixer from the population genome being evaluated
        with th.no_grad():
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        # Calculate TD error
        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask_expanded = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask_expanded

        # Mean TD error
        mean_td_error = masked_td_error.sum() / (mask.sum())
        

        return mean_td_error


    def calculate_adversarial_loss(self, batch, genome_index, population_attackers=None):
        """
        Fitness 2 (CORRECTED): Adversarial Q-Value Expectation (对抗价值期望)
        
        NEW MECHANISM (V7 - Critical Fix):
        计算防御者在受攻击情况下的 Q 值期望，优化目标是 **最大化** 对抗 Q 值。
        
        数学定义:
        J_2(θ) = E_s [ min_φ Q_tot(s, u_φ(s); θ) ]
        
        物理含义:
        - 不再追求"Q值稳定不变"（错误的指标）
        - 而是追求"在最坏攻击下，Q值依然尽可能高"（正确的对抗目标）
        - 筛选出那些"被攻击后依然认为自己能获得高回报"的防御者
        - 进化压力：Q值掉得太厉害的个体（认为自己死定了）会被淘汰
        
        Implementation:
        1. 从当前 batch 中获取受攻击的动作序列（已记录在 batch["byzantine_actions"] 中）
        2. 使用这些被攻击的动作计算 Q_tot（无需环境交互，完全离线）
        3. 对多个采样的攻击者取 min（最坏情况）
        4. 返回期望值（越大越好）
        
        Key Differences from Old Approach:
        - OLD: -|ΔQ_tot| / |ΔQ_k| (错误：追求数值稳定性)
        - NEW: E[min_φ Q_tot(attacked)] (正确：追求对抗价值最大化)
        
        Args:
            batch: EpisodeBatch containing experience data
                   Must have: actions, byzantine_actions, victim_id, state, obs
            genome_index: Index of the Genome in self.pop_genome to evaluate
            population_attackers: List of attacker instances for sampling (optional)
                                If None, uses batch's recorded attacker
            
        Returns:
            adversarial_q_value: Mean Q_tot under adversarial actions (higher is better)
        """
        # Get the Genome from population
        pop_genome = self.pop_genome[genome_index]
        
        # Get mask for valid timesteps
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # Set to eval mode
        pop_genome.eval()
        
        # === Step 1: Get adversarial actions from batch ===
        # These are the actions actually executed under attack (recorded during data collection)
        byzantine_actions = batch["byzantine_actions"][:, :-1]  # (batch, seq_len-1, n_agents, 1)
        victim_ids = batch["victim_id"][:, :-1]  # (batch, seq_len-1, 1)
        original_actions = batch["actions"][:, :-1]  # (batch, seq_len-1, n_agents, 1)
        
        # Construct adversarial action sequence: 
        # For each timestep, if victim_id < n_agents, use byzantine_action for that agent
        n_agents = self.args.n_agents
        batch_size = batch.batch_size
        seq_len = byzantine_actions.shape[1]
        
        adversarial_actions = original_actions.clone()
        for b in range(batch_size):
            for t in range(seq_len):
                victim = victim_ids[b, t, 0].item()
                victim = int(victim)
                if victim < n_agents:  # Valid attack (not "no-attack")
                    adversarial_actions[b, t, victim] = byzantine_actions[b, t, victim]
        
        # === Step 2: Compute Q-values under adversarial actions ===
        pop_genome.init_hidden(batch.batch_size)
        agent_q_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                agent_out = pop_genome.forward(batch, t=t)  # (bs, n_agents, n_actions)
                agent_q_list.append(agent_out)
        
        agent_q = th.stack(agent_q_list, dim=1)[:, :-1]  # (batch, seq_len-1, n_agents, n_actions)
        
        # Pick Q-values for adversarial actions
        adversarial_q = th.gather(agent_q, dim=3, index=adversarial_actions).squeeze(3)  # (batch, seq_len-1, n_agents)
        
        # === Step 3: Compute Global Q_tot under attack ===
        with th.no_grad():
            adversarial_q_tot = self.mixer(adversarial_q, batch["state"][:, :-1])  # (batch, seq_len-1, 1)
        
        # === Step 4: Apply mask and compute mean ===
        masked_q_tot = adversarial_q_tot * mask
        mean_adversarial_q = masked_q_tot.sum() / (mask.sum() + 1e-8)
        
        # === Optional: Sample multiple attackers and take min (worst-case) ===
        # If population_attackers is provided, we can compute Q_tot for multiple attack strategies
        # and take the minimum (most pessimistic estimate)
        if population_attackers is not None and len(population_attackers) > 0:
            num_attacker_samples = min(3, len(population_attackers))  # Sample 3 attackers
            sampled_attackers = np.random.choice(
                len(population_attackers), 
                size=num_attacker_samples, 
                replace=False
            )
            
            all_adversarial_q = [mean_adversarial_q]
            
            for attacker_idx in sampled_attackers:
                attacker = population_attackers[attacker_idx]
                attacker.eval()
                
                # Recompute adversarial actions using this attacker's strategy
                counterfactual_actions = original_actions.clone()
                
                with th.no_grad():
                    for t in range(seq_len + 1):  # +1 because we need full sequence
                        if t >= batch.max_seq_length:
                            break
                        attacker_logits = attacker.batch_forward(batch, t)  # (bs, n_agents+1)
                        victim_id = attacker_logits.argmax(dim=1)  # (bs,)
                        
                        for b in range(batch_size):
                            victim = victim_id[b].item()
                            if victim < n_agents and t < seq_len:
                                counterfactual_actions[b, t, victim] = byzantine_actions[b, t, victim]
                
                # Compute Q_tot for this attack strategy
                counterfactual_q = th.gather(agent_q, dim=3, index=counterfactual_actions).squeeze(3)
                with th.no_grad():
                    counterfactual_q_tot = self.mixer(counterfactual_q, batch["state"][:, :-1])
                
                masked_q = counterfactual_q_tot * mask
                mean_q = masked_q.sum() / (mask.sum() + 1e-8)
                all_adversarial_q.append(mean_q)
            
            # Take minimum across sampled attackers (worst-case robustness)
            mean_adversarial_q = th.stack(all_adversarial_q).min()
        
        # Return the adversarial Q-value (higher is better for defender)
        return mean_adversarial_q
    
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
                
        
        # Calculate estimated Q-Values for MAC in Genome
        self.Genome.train()  # Set mac to train mode
        
        # Forward pass for mac
        mac_out = []
        self.Genome.mac.init_hidden(batch.batch_size)
        
        for t in range(batch.max_seq_length):
            agent_outs = self.Genome.mac.forward(batch, t=t)
            mac_out.append(agent_outs)
            
        mac_out = th.stack(mac_out, dim=1)  # (batch, seq_len, n_agents, n_actions)
        
        # Pick the Q-Values for the actions taken by each agent
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)

        # Calculate the Q-Values necessary for the target
        with th.no_grad():
            self.target_Genome.train()  # Set target Genome to train mode
            target_mac_out = []
            self.target_Genome.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                target_agent_outs = self.target_Genome.forward(batch, t=t)
                target_mac_out.append(target_agent_outs)

            target_mac_out = th.stack(target_mac_out, dim=1)  # Concat across time

            # Max over target Q-Values
            mac_out_detach = mac_out.clone().detach()
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

        # Mixer
        chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        # Calculate TD error
        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask

        # important sampling for PER
        if self.use_per:
            per_weight = th.from_numpy(per_weight).unsqueeze(-1).to(device=self.device)
            masked_td_error = masked_td_error.sum(1) * per_weight

        # Calculate loss
        L_td = masked_td_error.sum() / mask.sum()
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
            self.logger.log_stat("td_error_abs", (masked_td_error.abs().sum().item()/mask_elems), t_env)
            self.logger.log_stat("q_taken_mean", (chosen_action_qvals * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            self.logger.log_stat("target_mean", (targets * mask).sum().item()/(mask_elems * self.args.n_agents), t_env)
            
            self.log_stats_t = t_env
            
            # print estimated matrix
            if self.args.env == "one_step_matrix_game":
                print_matrix_status(batch, self.mixer, mac_out)

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
                info["td_errors_abs"] = ((td_error.abs() * mask).sum(1) \
                                / th.sqrt(mask.sum(1))).detach().to('cpu')
        
        # Store TD error for learning-assisted dynamic weighting
        self.last_td_error = L_td.item()
        info["td_error"] = self.last_td_error
        
        return info
    
    def _update_targets(self):
        self.target_Genome.load_state(self.Genome)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.Genome.cuda()
        self.target_Genome.cuda()
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
            
            # Calculate Q-Values using mac
            mac_out = []
            self.Genome.mac.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.Genome.mac.forward(batch, t=t)
                mac_out.append(agent_outs)
            mac_out = th.stack(mac_out, dim=1)
        
        # Pick Q-Values for the COUNTERFACTUAL actions (attacker-perturbed actions)
        # This is the key: different attackers → different victim selections → different counterfactual actions
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions_counterfactual).squeeze(3)
        
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
            mac_out_detach = mac_out.clone().detach()
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
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])

        # Calculate TD error
        td_error = (chosen_action_qvals - targets.detach())
        td_error2 = 0.5 * td_error.pow(2)

        mask_expanded = mask.expand_as(td_error2)
        masked_td_error = td_error2 * mask_expanded

        # Mean TD error
        mean_td_error = masked_td_error.sum() / (mask_expanded.sum() + 1e-8)
        
        # Safety check: if TD error is unreasonably large, cap it
        if mean_td_error.item() > 1e6:
            return th.tensor(1e6, device=mean_td_error.device)
        
        # Return TD error (higher = better for attacker, as it means more disruption)
        return mean_td_error
    

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
    def _get_attacker_conditional_victim_distribution(self, attacker, batch):
        """
        NEW (MACO QD): Extract Conditional Victim Distribution as Behavior Descriptor.
        
        Behavior Vector V_φ = [p_1, p_2, ..., p_N]:
        - p_i = Expected probability of attacking agent i (excluding No-Op)
        - Uses softmax probabilities to capture stochastic attack preferences
        
        Key Insight:
        - Uses expected probabilities (not greedy sampling) for stable behavior descriptors
        - Ignores "No-Op" action (index N) by marginalizing over victim actions
        - Forms a probability simplex over N agents (not N+1)
        
        Returns:
            victim_dist: (N,) numpy array - Conditional probability over victims
        """
        with th.no_grad():
            bs = batch.batch_size
            seq_len = batch.max_seq_length
            n_agents = self.args.n_agents
            
            # FIXED (Bug 4): Use expected probabilities instead of greedy sampling
            # Method: Compute softmax probabilities and average over time
            all_probs_agents = []
            for t in range(seq_len):
                attacker_logits = attacker.batch_forward(batch, t)  # (bs, n_agents+1)
                attack_probs = th.softmax(attacker_logits, dim=-1)  # (bs, n_agents+1)
                attack_probs_agents_only = attack_probs[:, :n_agents]  # (bs, n_agents) - exclude No-Op
                all_probs_agents.append(attack_probs_agents_only)
            
            all_probs_agents = th.stack(all_probs_agents, dim=1)  # (bs, seq_len, n_agents)
            
            # Expected victim distribution: average probability across batch and time
            victim_dist = all_probs_agents.mean(dim=(0, 1))  # (n_agents,)
            
            # Renormalize (in case of numerical errors)
            victim_dist_sum = victim_dist.sum()
            if victim_dist_sum > 1e-8:
                victim_dist = victim_dist / victim_dist_sum
            else:
                # Edge case: all probabilities near zero → uniform distribution
                victim_dist = th.ones(n_agents, device=self.device) / n_agents
            
            return victim_dist.cpu().numpy()  # (N,)
    
    def _get_attacker_action_distribution_batch(self, attacker, batch):
        """
        DEPRECATED (kept for backward compatibility).
        
        OLD MECHANISM: Get action probability distribution for an attacker over ALL timesteps.
        This function is replaced by _get_attacker_conditional_victim_distribution()
        in the new QD framework.
        
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
        REFACTORED V2: Memetic SGD for Attacker using Counterfactual Attack Advantage.
        
        New Training Mechanism (MACO QD):
        1. Attack Utility: U_att(j|s) = Q_tot(s, u) - Q_tot(s, <u_worst_j, u_-j>)
           - Measures the counterfactual drop in global Q when agent j is attacked
        2. Budget Modulation: γ(k) = 1 + λ(K-k)/K (higher threshold when budget is scarce)
        3. Attack Advantage: A_att(j|s,k) = U_att(j|s) - γ(k)·Ū(s)
        4. Policy Gradient Loss: L_att = -E[Σ A_att(j)·log π(j)]
        
        Key Innovation:
        - Counterfactual reasoning: directly measures global value disruption
        - Budget-aware opportunity cost: forces strategic target selection
        - No environment interaction: purely offline based on defender's Q_tot
        
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
            
            # === Step 2: Compute Counterfactual Attack Utility (OPTIMIZED: Batch Computation) ===
            # For each agent j, compute: U_att(j) = Q_tot(u) - Q_tot(<u_worst_j, u_-j>)
            
            # 2.1: Get Q_tot for normal actions
            with th.no_grad():
                Q_tot_normal = self.mixer(chosen_q, batch["state"][:, :-1])  # (bs, seq_len-1, 1)
            
            # 2.2: Batch compute Q_tot for ALL counterfactual scenarios at once
            n_agents = self.args.n_agents
            bs, seq_len, _ = chosen_q.shape
            
            # FIXED (Bug 2): Add dimension validation
            assert batch["state"].shape[1] == batch.max_seq_length, \
                f"State dim mismatch: {batch['state'].shape[1]} vs {batch.max_seq_length}"
            assert batch["state"][:, :-1].shape[1] == seq_len, \
                f"State sequence mismatch: {batch['state'][:, :-1].shape[1]} vs {seq_len}"
            
            # Find worst action for each agent: argmin Q_j(o_j, a)
            worst_actions = agent_q.argmin(dim=3)  # (bs, seq_len-1, n_agents)
            
            # Get worst Q values for all agents
            worst_q_values = th.gather(agent_q, dim=3, index=worst_actions.unsqueeze(-1)).squeeze(-1)  # (bs, seq_len-1, n_agents)
            
            # OPTIMIZED: Create diagonal mask to replace each agent's Q with worst Q one at a time
            # Shape: (n_agents, bs, seq_len-1, n_agents)
            # For scenario j: all agents use chosen_q except agent j uses worst_q_values
            counterfactual_q_all = chosen_q.unsqueeze(0).expand(n_agents, -1, -1, -1).clone()  # (n_agents, bs, seq_len-1, n_agents)
            
            # Create index tensor for advanced indexing
            agent_indices = th.arange(n_agents, device=self.device)
            counterfactual_q_all[agent_indices, :, :, agent_indices] = worst_q_values.permute(2, 0, 1)  # Replace diagonal
            
            # Reshape for batch mixer computation: (n_agents * bs, seq_len-1, n_agents)
            counterfactual_q_batch = counterfactual_q_all.reshape(n_agents * bs, seq_len, n_agents)
            state_batch = batch["state"][:, :-1].unsqueeze(0).expand(n_agents, -1, -1, -1).reshape(n_agents * bs, seq_len, -1)
            
            # Single mixer call for all counterfactual scenarios!
            with th.no_grad():
                Q_tot_counterfactual_batch = self.mixer(counterfactual_q_batch, state_batch)  # (n_agents * bs, seq_len-1, 1)
            
            # Reshape back: (n_agents, bs, seq_len-1, 1)
            Q_tot_counterfactual = Q_tot_counterfactual_batch.reshape(n_agents, bs, seq_len, 1)
            
            # Compute attack utilities: (n_agents, bs, seq_len-1) -> transpose to (bs, seq_len-1, n_agents)
            attack_utilities = (Q_tot_normal.unsqueeze(0) - Q_tot_counterfactual).squeeze(-1).permute(1, 2, 0)
            
            # Ensure non-negative (due to monotonicity, should be ≥ 0)
            attack_utilities = th.clamp(attack_utilities, min=0.0)  # (bs, seq_len-1, n_agents)
            
            # FIXED (Bug 3): Check for NaN in attack_utilities
            if not th.isfinite(attack_utilities).all():
                print("[WARNING] NaN detected in attack_utilities")
                return float('inf')
            
            # === Step 3: Compute Budget-Aware Attack Advantage ===
            # Baseline utility: average attack utility across all agents
            U_bar = attack_utilities.mean(dim=2, keepdim=True)  # (bs, seq_len-1, 1)
            
            # Budget modulation factor: γ(k) = 1 + λ·(K-k)/K
            # As remaining budget k decreases → (K-k) increases → γ increases (higher threshold)
            gamma_k = 1.0 + budget_lambda * (K - left_attack) / K  # (bs, seq_len-1, 1)
            
            # Attack Advantage: A_att(j) = U_att(j) - γ(k)·Ū
            attack_advantage = attack_utilities - gamma_k * U_bar  # (bs, seq_len-1, n_agents)
            
            # For "no-attack" option (j=N+1), advantage is 0 (baseline anchor)
            no_attack_advantage = th.zeros_like(attack_advantage[:, :, :1])  # (bs, seq_len-1, 1)
            attack_advantage_full = th.cat([attack_advantage, no_attack_advantage], dim=2)  # (bs, seq_len-1, n_agents+1)
            
            # === Step 4: Get attacker's action probabilities ===
            attacker_logits = []
            for t in range(batch.max_seq_length):
                logit = attacker.batch_forward(batch, t=t)
                attacker_logits.append(logit)
            attacker_logits = th.stack(attacker_logits, dim=1)[:, :-1]  # (bs, seq_len-1, n_agents+1)
            
            # Convert to log probabilities
            log_probs = th.log_softmax(attacker_logits, dim=-1)  # (bs, seq_len-1, n_agents+1)
            
            # FIXED (Bug 3): Check for NaN in log_probs
            if not th.isfinite(log_probs).all():
                print("[WARNING] NaN detected in log_probs")
                return float('inf')
            
            # === Step 5: Policy Gradient Loss ===
            # L_att = -E[Σ_j A_att(j) · log π(j)]
            # If A_att(j) > 0: High advantage → increase log π(j)  
            # If A_att(j) < 0 (all below baseline): advantage of No-Op (0) becomes highest
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
    
    
    # ============================================================================
    # NEW: Quality-Diversity (QD) Framework for Attacker Evolution
    # ============================================================================
    
    def quality_diversity_selection(self, batch, population_attackers, num_clusters=None):
        """
        NEW (MACO QD): Quality-Diversity Selection using K-Means Clustering.
        
        Replaces NSGA-II dual-objective selection with explicit behavior space partitioning.
        
        Algorithm:
        1. Compute fitness (TD Error) for each attacker
        2. Extract behavior vectors V_φ (Conditional Victim Distribution)
        3. Cluster behavior space into K clusters via K-Means
        4. Select best attacker (max TD Error) from each cluster
        
        Key Innovation:
        - Explicit behavior diversity via clustering (not implicit Pareto front)
        - Fitness is ONLY TD Error (single objective)
        - Diversity is enforced by K-Means partitioning
        - Ensures exploration of different attack strategies
        
        Args:
            batch: Episode batch for fitness evaluation
            population_attackers: List of attacker instances
            num_clusters: Number of clusters (default: pop_size // 2)
            
        Returns:
            elite_indices: Indices of selected attackers (one per cluster)
            replace_indices: Indices of attackers to be replaced
            fitness_dict: Dictionary with fitness and behavior info
        """
        from sklearn.cluster import KMeans
        
        pop_size = len(population_attackers)
        n_agents = self.args.n_agents
        
        # Default: half of population becomes elites
        if num_clusters is None:
            num_clusters = max(1, pop_size // 2)
        
        # === Step 1: Compute TD Error (Quality) for all attackers ===
        td_errors = []
        for i in range(pop_size):
            with th.no_grad():
                td_error = self.calculate_attacker_TD_error(
                    batch, i, population_attackers
                )
            td_errors.append(td_error.item())
        
        td_errors = np.array(td_errors)
        
        # === Step 2: Extract Behavior Vectors (Diversity) - with progress tracking ===
        behavior_vectors = []
        for i, attacker in enumerate(population_attackers):
            victim_dist = self._get_attacker_conditional_victim_distribution(attacker, batch)
            behavior_vectors.append(victim_dist)
        
        behavior_vectors = np.array(behavior_vectors)  # (pop_size, N)
        
        # === Step 3: K-Means Clustering in Behavior Space (OPTIMIZED) ===
        # FIXED (Bug 8): Check if behavior vectors are too similar (early training)
        behavior_variance = np.var(behavior_vectors, axis=0).sum()
        
        if behavior_variance < 1e-6:
            # All attackers have nearly identical behavior → random selection
            print(f"[QD Selection WARNING] Behavior vectors too similar (var={behavior_variance:.2e}), using random selection")
            elite_indices = np.random.choice(pop_size, size=min(num_clusters, pop_size), replace=False).tolist()
            replace_indices = [i for i in range(pop_size) if i not in elite_indices]
            
            # Create dummy cluster labels
            cluster_labels = np.zeros(pop_size, dtype=int)
            for i, elite_idx in enumerate(elite_indices):
                cluster_labels[elite_idx] = i
            
            fitness_dict = {
                'td_errors': td_errors,
                'behavior_vectors': behavior_vectors,
                'cluster_labels': cluster_labels,
                'elite_indices': elite_indices,
                'replace_indices': replace_indices,
                'warning': 'behavior_vectors_too_similar'
            }
            
            return elite_indices, replace_indices, fitness_dict
        
        # Handle edge cases
        if num_clusters >= pop_size:
            # Each attacker is its own cluster
            cluster_labels = np.arange(pop_size)
        else:
            # Optimized K-Means parameters:
            # - n_init='auto' (faster, sklearn 1.4+) or 3 (legacy compatibility)
            # - max_iter=100 (reduced from default 300)
            # - tol=1e-3 (relaxed from default 1e-4)
            try:
                kmeans = KMeans(
                    n_clusters=num_clusters, 
                    random_state=42, 
                    n_init='auto',  # sklearn 1.4+: adaptive initialization
                    max_iter=100,    # Reduced iterations
                    tol=1e-3         # Relaxed tolerance
                )
            except TypeError:
                # Fallback for older sklearn versions
                kmeans = KMeans(
                    n_clusters=num_clusters, 
                    random_state=42, 
                    n_init=3,        # Reduced from default 10
                    max_iter=100, 
                    tol=1e-3
                )
            cluster_labels = kmeans.fit_predict(behavior_vectors)
        
        # === Step 4: Select Best Attacker from Each Cluster ===
        elite_indices = []
        for cluster_id in range(num_clusters):
            # Get all attackers in this cluster
            cluster_mask = (cluster_labels == cluster_id)
            cluster_attackers = np.where(cluster_mask)[0]
            
            # FIXED (Bug 6): Handle empty clusters by selecting from existing elites
            if len(cluster_attackers) == 0:
                if len(elite_indices) > 0:
                    # Fill with random existing elite to maintain num_clusters elites
                    elite_indices.append(np.random.choice(elite_indices))
                continue
            
            # Select attacker with highest TD Error in this cluster
            cluster_td_errors = td_errors[cluster_attackers]
            best_in_cluster = cluster_attackers[np.argmax(cluster_td_errors)]
            elite_indices.append(int(best_in_cluster))
        
        # === Step 5: Determine Replace Indices ===
        elite_set = set(elite_indices)
        replace_indices = [i for i in range(pop_size) if i not in elite_set]
        
        # === Step 6: Return Results ===
        fitness_dict = {
            'td_errors': td_errors,
            'behavior_vectors': behavior_vectors,
            'cluster_labels': cluster_labels,
            'elite_indices': elite_indices,
            'replace_indices': replace_indices
        }
        
        return elite_indices, replace_indices, fitness_dict
    

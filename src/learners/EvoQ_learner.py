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


class EvoQLearner:
    def __init__(self, genome, population, scheme, logger, args):
        self.args = args
        self.Genome = genome  # Genome contains two MACs (mac1 and mac2)
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
            
            
        # Evolution: Create population MACs
        self.pop_size = args.pop_size
        self.elite_size = args.elite_size
        self.population = population
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
    
    
    def calculate_TD_error(self, batch: EpisodeBatch, mac_index: int):
        """
        Calculate TD error for a specific MAC in the population.
        Exactly follows the TD error computation logic in train() function.
        
        Args:
            batch: EpisodeBatch containing experience data
            mac_index: Index of the MAC in self.population to evaluate
            
        Returns:
            mean_td_error: Mean TD error for the specified MAC (scalar tensor)
        """
        # Get the Genome from population (contains mac1 and mac2)
        pop_genome = self.population[mac_index]
        
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

        # Sum TD errors from both MACs
        mean_td_error_1 = masked_td_error_1.sum() / mask_expanded.sum()
        mean_td_error_2 = masked_td_error_2.sum() / mask_expanded.sum()
        mean_td_error = mean_td_error_1 + mean_td_error_2
        
        return mean_td_error


    def calculate_confidence_Q(self, batch: EpisodeBatch, mac_index: int):
        """
        Calculate confidence Q metric for a specific MAC in the population.
        Exactly follows the confidence Q computation logic in train() function.
        
        Args:
            batch: EpisodeBatch containing experience data
            mac_index: Index of the MAC in self.population to evaluate
            
        Returns:
            mean_confidence_Q: Mean confidence Q for the specified MAC (scalar tensor)
        """
        # Get the MAC from population
        pop_mac = self.population[mac_index]
        
        # Get the relevant quantities (same as in train())
        rewards = batch["reward"][:, :-1]
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        avail_actions = batch["avail_actions"]
        
        # Calculate estimated Q-Values using the population MAC
        pop_mac.eval()  # Set to eval mode for evaluation
        mac_out = []
        pop_mac.init_hidden(batch.batch_size)
        
        with th.no_grad():  # No gradients needed for evaluation
            for t in range(batch.max_seq_length):
                agent_outs = pop_mac.forward(batch, t=t)
                mac_out.append(agent_outs)
        
        mac_out = th.stack(mac_out, dim=1)  # (batch, seq_len, n_agents, n_actions)
        
        chosen_action_qvals = th.gather(mac_out[:, :-1], dim=3, index=actions).squeeze(3)
        # Calculate mean confidence Q 
        with th.no_grad():
            chosen_action_qvals = self.mixer(chosen_action_qvals, batch["state"][:, :-1])
        mean_confidence_Q = (chosen_action_qvals * mask).sum() / mask.sum()

        return mean_confidence_Q


    def calculate_adversarial_loss(self, batch: EpisodeBatch, mac_index: int):
        """
        Calculate robustness metrics for a specific MAC in the population.
        Evaluates robustness at the Global Q-value level (after Mixer).
        
        Args:
            batch: EpisodeBatch containing experience data
            mac_index: Index of the MAC in self.population to evaluate
            
        Returns:
            adv_loss: Mean adversarial loss for the specified MAC (scalar tensor)
        """
        # Get the Genome from population
        pop_genome = self.population[mac_index]
        # Get the relevant quantities for robust regularizer
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) # whether the sample is valid
        
        # Set both agents to eval mode for evaluation
        pop_genome.eval()
        adv_batch = noise_atk(batch, self.args)
        normal_global_q = []
        adv_global_q = []
        
        # compute normal Q
        pop_genome.init_hidden(batch.batch_size)
        with th.no_grad():  # No gradients needed for evaluation
            for t in range(batch.max_seq_length):  # Exclude last timestep
                # Normal Q-values from Genome (min of two MACs)
                agent_out = pop_genome.forward(batch, t=t)  # (batch, n_agents, n_actions)
                normal_global_q.append(agent_out)
        
        # compute adversarial Q
        pop_genome.init_hidden(batch.batch_size)
        adv_batch = noise_atk(batch, self.args)
        with th.no_grad():
            # Create adversarial batch
            for t in range(batch.max_seq_length):  # Exclude last timestep
                # Adversarial Q-values from Genome
                adv_agent_out = pop_genome.forward(adv_batch, t=t)
                adv_global_q.append(adv_agent_out)
        
        # Stack and take mean over the last dimension (n_actions), then remove it
        normal_global_q = th.stack(normal_global_q, dim=1)  # (batch, seq_len, n_agents, num_actions)
        adv_global_q = th.stack(adv_global_q, dim=1) # (batch, seq_len, n_agents, num_actions)

        # Apply mixer to get Global Q
        with th.no_grad():
            normal_global_q = self.mixer(normal_global_q, batch["state"])[:, :-1] * mask # (batch, seq_len, 1)
            adv_global_q = self.mixer(adv_global_q, adv_batch["state"])[:, :-1] * mask # (batch, seq_len, 1)
                
        
        # Calculate the difference in Global Q-values
        return get_diff(normal_global_q, adv_global_q).sum() / mask.sum()

    def calculate_adversarial_entropy(self, batch, mac_index):
        """
        Calculate adversarial entropy for robustness evaluation.
        
        Entropy measures the uncertainty/confusion in the policy's decision distribution
        under adversarial perturbations. Lower entropy = more certain/confident decision.
        
        This is the foundation metric (计算熵), while the caller interprets it as
        "robustness confidence" (鲁棒自信度) by taking the negative.
        
        Symmetry with other metrics:
        - Optimality side: TD Error (accuracy) + Q Value (ambition)
        - Robustness side: Q Smoothness (stability) + Entropy (certainty foundation)
        
        Args:
            batch: EpisodeBatch containing experience data
            mac_index: Index of the MAC in self.population to evaluate
            
        Returns:
            neg_entropy: Negative entropy (higher = more confident/robust)
        """
        # Get the Genome from population
        pop_genome = self.population[mac_index]
        
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
                # Get Q-values for all actions from Genome (min of mac1 and mac2)
                agent_out = pop_genome.forward(adv_batch, t=t)  # (batch, n_agents, n_actions)
                adv_global_q_list.append(agent_out)
        
        # Stack: (batch, seq_len, n_agents, n_actions)
        adv_agent_q = th.stack(adv_global_q_list, dim=1)
        
        # Apply mixer to get Global Q for all actions
        # mixer handles 4D input: (batch, seq_len, n_agents, n_actions) -> (batch, seq_len-1, n_actions)
        with th.no_grad():
            adv_global_q = self.mixer(adv_agent_q, adv_batch["state"])[:, :-1]  # (batch, seq_len-1, n_actions)
        
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
    
    def calculate_twin_adversarial_entropy(self, batch, mac_index):
        """
        Enhanced version: Twin Adversarial Entropy.
        
        Computes a combined metric of entropy from both Q-networks plus their disagreement.
        Lower value = both networks are confident AND in agreement.
        
        This is the foundation metric (计算熵+分歧), while the caller interprets it as
        "twin robustness confidence" (双重鲁棒自信度) by taking the negative.
        
        Fitness = -(H(π1) + H(π2) + λ·KL(π1||π2))
        
        This ensures both networks are:
        1. Individually confident (low entropy)
        2. In agreement with each other (low KL divergence)
        
        Args:
            batch: EpisodeBatch containing experience data
            mac_index: Index of the MAC in self.population to evaluate
            
        Returns:
            twin_conf: Negative of (entropy1 + entropy2 + λ·KL divergence)
        """
        # Get the Genome from population
        pop_genome = self.population[mac_index]
        
        # Get mask for valid timesteps
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # Set to eval mode
        pop_genome.eval()
        
        # Create adversarial observations
        adv_batch = noise_atk(batch, self.args)
        
        # Compute Global Q-values from both MACs
        pop_genome.mac1.init_hidden(batch.batch_size)
        pop_genome.mac2.init_hidden(batch.batch_size)
        
        adv_q1_list = []
        adv_q2_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                # Q-values from mac1
                agent_out1 = pop_genome.mac1.forward(adv_batch, t=t)  # (batch, n_agents, n_actions)
                adv_q1_list.append(agent_out1)
                
                # Q-values from mac2
                agent_out2 = pop_genome.mac2.forward(adv_batch, t=t)  # (batch, n_agents, n_actions)
                adv_q2_list.append(agent_out2)
        
        # Stack: (batch, seq_len, n_agents, n_actions)
        adv_agent_q1 = th.stack(adv_q1_list, dim=1)
        adv_agent_q2 = th.stack(adv_q2_list, dim=1)
        
        # Apply mixer to get Global Q for all actions
        with th.no_grad():
            adv_global_q1 = self.mixer(adv_agent_q1, adv_batch["state"])[:, :-1]  # (batch, seq_len-1, n_actions)
            adv_global_q2 = self.mixer(adv_agent_q2, adv_batch["state"])[:, :-1]  # (batch, seq_len-1, n_actions)
        
        # Compute Boltzmann distributions
        tau = getattr(self.args, 'adversarial_tau', 1.0)
        
        log_probs1 = th.nn.functional.log_softmax(adv_global_q1 / tau, dim=-1)
        log_probs2 = th.nn.functional.log_softmax(adv_global_q2 / tau, dim=-1)
        probs1 = th.exp(log_probs1)
        probs2 = th.exp(log_probs2)
        
        # Compute individual entropies
        entropy1 = -(probs1 * log_probs1).sum(dim=-1)  # (batch, seq_len-1)
        entropy2 = -(probs2 * log_probs2).sum(dim=-1)  # (batch, seq_len-1)
        
        # Compute KL divergence: KL(π1||π2) = Σ π1(a) log(π1(a)/π2(a))
        kl_div = (probs1 * (log_probs1 - log_probs2)).sum(dim=-1)  # (batch, seq_len-1)
        
        # Weight for KL term
        kl_weight = getattr(self.args, 'kl_weight', 0.5)
        
        # Combined metric: H1 + H2 + λ·KL (all positive, higher = worse)
        combined_metric = entropy1 + entropy2 + kl_weight * kl_div  # (batch, seq_len-1)
        
        # Apply mask and compute mean
        masked_metric = combined_metric * mask
        mean_metric = masked_metric.sum() / mask.sum()
        
        # Return the combined entropy+divergence (caller will negate for "confidence")
        return mean_metric

    def calculate_evolutionary_consensus(self, batch, mac_index, elite_indices):
        """
        Calculate Evolutionary Consensus Score (TEVC-native metric).
        
        Combines:
        1. Internal Consistency: KL divergence between Twin Q-networks
        2. Evolutionary Consensus: KL divergence between individual and population ensemble
        
        This metric leverages the unique advantage of evolutionary algorithms - 
        having access to multiple elite individuals for ensemble-based robustness.
        
        Fitness = -(KL(π_Q1||π_Q2) + β·KL(π_me||π_ensemble))
        
        Args:
            batch: EpisodeBatch containing experience data
            mac_index: Index of current individual to evaluate
            elite_indices: List of indices of elite individuals in population
            
        Returns:
            consensus_score: Negative of combined KL divergence (higher = better consensus)
        """
        # Get the Genome from population
        pop_genome = self.population[mac_index]
        
        # Get mask for valid timesteps
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # Set to eval mode
        pop_genome.eval()
        
        # Create adversarial observations
        adv_batch = noise_atk(batch, self.args)
        
        # === Part 1: Internal Consistency (Twin-Q Networks) ===
        pop_genome.mac1.init_hidden(batch.batch_size)
        pop_genome.mac2.init_hidden(batch.batch_size)
        
        adv_q1_list = []
        adv_q2_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                # Q-values from mac1
                agent_out1 = pop_genome.mac1.forward(adv_batch, t=t)  # (batch, n_agents, n_actions)
                adv_q1_list.append(agent_out1)
                
                # Q-values from mac2
                agent_out2 = pop_genome.mac2.forward(adv_batch, t=t)  # (batch, n_agents, n_actions)
                adv_q2_list.append(agent_out2)
        
        # Stack and apply mixer
        adv_agent_q1 = th.stack(adv_q1_list, dim=1)
        adv_agent_q2 = th.stack(adv_q2_list, dim=1)
        
        with th.no_grad():
            adv_global_q1 = self.mixer(adv_agent_q1, adv_batch["state"])[:, :-1]  # (batch, seq_len-1, n_actions)
            adv_global_q2 = self.mixer(adv_agent_q2, adv_batch["state"])[:, :-1]  # (batch, seq_len-1, n_actions)
        
        # Compute Boltzmann distributions
        tau = getattr(self.args, 'adversarial_tau', 1.0)
        
        log_probs1 = th.nn.functional.log_softmax(adv_global_q1 / tau, dim=-1)
        log_probs2 = th.nn.functional.log_softmax(adv_global_q2 / tau, dim=-1)
        probs1 = th.exp(log_probs1)
        probs2 = th.exp(log_probs2)
        
        # KL divergence between twin networks: KL(π1||π2)
        kl_internal = (probs1 * (log_probs1 - log_probs2)).sum(dim=-1)  # (batch, seq_len-1)
        
        # === Part 2: Evolutionary Consensus (Population Ensemble) ===
        # Build ensemble policy from elite individuals
        K = min(len(elite_indices), getattr(self.args, 'ensemble_size', 3))  # Top-K elites
        if K == 0 or mac_index in elite_indices[:K]:
            # If no elites or evaluating an elite itself, skip ensemble term
            kl_ensemble = th.zeros_like(kl_internal)
        else:
            # Sample K elite individuals (excluding self if present)
            selected_elites = [idx for idx in elite_indices[:K] if idx != mac_index][:K]
            
            if len(selected_elites) == 0:
                kl_ensemble = th.zeros_like(kl_internal)
            else:
                # Collect Q-values from elite individuals
                elite_global_qs = []
                for elite_idx in selected_elites:
                    elite_genome = self.population[elite_idx]
                    elite_genome.eval()
                    elite_genome.init_hidden(batch.batch_size)
                    
                    elite_q_list = []
                    with th.no_grad():
                        for t in range(batch.max_seq_length):
                            elite_out = elite_genome.forward(adv_batch, t=t)
                            elite_q_list.append(elite_out)
                    
                    elite_agent_q = th.stack(elite_q_list, dim=1)
                    with th.no_grad():
                        elite_global_q = self.mixer(elite_agent_q, adv_batch["state"])[:, :-1]
                    elite_global_qs.append(elite_global_q)
                
                # Ensemble: average Q-values from elites
                ensemble_q = th.stack(elite_global_qs, dim=0).mean(dim=0)  # (batch, seq_len-1, n_actions)
                
                # Compute ensemble policy distribution
                log_probs_ensemble = th.nn.functional.log_softmax(ensemble_q / tau, dim=-1)
                probs_ensemble = th.exp(log_probs_ensemble)
                
                # Use mean of individual's two networks for comparison
                probs_me = (probs1 + probs2) / 2.0
                log_probs_me = th.log(probs_me + 1e-8)
                
                # KL divergence between individual and ensemble: KL(π_me||π_ensemble)
                kl_ensemble = (probs_me * (log_probs_me - log_probs_ensemble)).sum(dim=-1)  # (batch, seq_len-1)
        
        # === Combine both terms ===
        beta = getattr(self.args, 'ensemble_consensus_weight', 0.5)  # Weight for ensemble term
        
        combined_kl = kl_internal + beta * kl_ensemble  # (batch, seq_len-1)
        
        # Apply mask and compute mean
        masked_kl = combined_kl * mask.unsqueeze(-1)
        mean_kl = masked_kl.sum() / mask.sum()
        
        # Return negative (higher consensus = lower KL = higher fitness)
        return -mean_kl
    
    def calculate_adversarial_novelty(self, batch, mac_index):
        """
        Calculate Adversarial Behavioral Novelty (Quality-Diversity metric).
        
        Measures how different an individual's behavior is compared to historical elites
        under adversarial perturbations. This encourages diversity in robustness strategies.
        
        Novelty = avg_distance(π_me(·|o_adv), π_archive(·|o_adv))
        
        This metric promotes:
        1. Behavioral diversity: Different robust strategies, not just one
        2. Quality-Diversity: Robustness through diverse defensive patterns
        3. Attack resistance: Harder to find universal attacks
        
        Args:
            batch: EpisodeBatch containing experience data
            mac_index: Index of current individual to evaluate
            
        Returns:
            novelty_score: Average behavioral distance to archive (higher = more novel)
        """
        # Get the Genome from population
        pop_genome = self.population[mac_index]
        
        # Get mask for valid timesteps
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
        # Set to eval mode
        pop_genome.eval()
        
        # Create adversarial observations
        adv_batch = noise_atk(batch, self.args)
        
        # === Compute current individual's behavioral descriptor ===
        pop_genome.init_hidden(batch.batch_size)
        adv_q_list = []
        
        with th.no_grad():
            for t in range(batch.max_seq_length):
                agent_out = pop_genome.forward(adv_batch, t=t)  # (batch, n_agents, n_actions)
                adv_q_list.append(agent_out)
        
        # Stack and apply mixer
        adv_agent_q = th.stack(adv_q_list, dim=1)
        with th.no_grad():
            adv_global_q = self.mixer(adv_agent_q, adv_batch["state"])[:, :-1]  # (batch, seq_len-1, n_actions)
        
        # Compute action distribution (Boltzmann policy)
        tau = getattr(self.args, 'adversarial_tau', 1.0)
        probs_me = th.nn.functional.softmax(adv_global_q / tau, dim=-1)  # (batch, seq_len-1, n_actions)
        
        # Behavioral descriptor: mean action distribution over time
        # Shape: (batch, n_actions)
        behavior_me = probs_me.mean(dim=1)  # Average over time dimension
        
        # === Calculate novelty against archive ===
        if len(self.elite_archive) == 0:
            # Empty archive: maximum novelty (encourage exploration)
            novelty_score = th.tensor(1.0, device=self.device)
        else:
            # Compute distance to each archived behavior
            distances = []
            for archived_behavior in self.elite_archive:
                # archived_behavior: (batch, n_actions) tensor
                # Use JS divergence (symmetrized KL) as distance metric
                dist = self._js_divergence(behavior_me, archived_behavior)
                distances.append(dist)
            
            # Stack distances: (num_archived, batch)
            distances_tensor = th.stack(distances, dim=0)
            
            # Novelty = average distance to K-nearest neighbors
            K = min(self.novelty_k_nearest, len(self.elite_archive))
            
            # For each batch element, find K-nearest neighbors
            # Shape: (batch, num_archived)
            distances_transposed = distances_tensor.t()
            
            # Get K smallest distances for each batch element
            knn_distances, _ = th.topk(distances_transposed, K, dim=1, largest=False)
            
            # Novelty = mean of K-nearest distances
            novelty_per_sample = knn_distances.mean(dim=1)  # (batch,)
            
            # Apply mask and average over batch
            novelty_score = (novelty_per_sample * mask.mean(dim=1)).sum() / (mask.mean(dim=1).sum() + 1e-8)
        
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
        Update the elite archive with behavioral descriptors from current elites.
        
        Args:
            elite_indices: List of indices of elite individuals
            batch: EpisodeBatch for computing behavioral descriptors
        """
        # Create adversarial batch
        adv_batch = noise_atk(batch, self.args)
        tau = getattr(self.args, 'adversarial_tau', 1.0)
        
        # Compute behavioral descriptors for each elite
        for elite_idx in elite_indices[:min(len(elite_indices), 5)]:  # Add top-5 elites
            elite_genome = self.population[elite_idx]
            elite_genome.eval()
            elite_genome.init_hidden(batch.batch_size)
            
            adv_q_list = []
            with th.no_grad():
                for t in range(batch.max_seq_length):
                    agent_out = elite_genome.forward(adv_batch, t=t)
                    adv_q_list.append(agent_out)
            
            adv_agent_q = th.stack(adv_q_list, dim=1)
            with th.no_grad():
                adv_global_q = self.mixer(adv_agent_q, adv_batch["state"])[:, :-1]
            
            # Behavioral descriptor: mean action distribution
            probs = th.nn.functional.softmax(adv_global_q / tau, dim=-1)
            behavior = probs.mean(dim=1)  # (batch, n_actions)
            
            # Add to archive (detach and clone to avoid memory issues)
            self.elite_archive.append(behavior.detach().clone())
        
        # Maintain archive size (FIFO)
        if len(self.elite_archive) > self.archive_max_size:
            # Remove oldest entries
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

    def _update_targets(self):
        self.target_Genome.load_state(self.Genome)
        if self.mixer is not None:
            self.target_mixer.load_state_dict(self.mixer.state_dict())
        self.logger.console_logger.info("Updated target network")

    def cuda(self):
        self.Genome.cuda()
        self.target_Genome.cuda()
        for pop_genome in self.population:
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

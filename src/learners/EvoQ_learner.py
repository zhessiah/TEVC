import copy
from components.episode_buffer import EpisodeBatch
from modules.mixers.nmix import Mixer
from modules.mixers.vdn import VDNMixer
from modules.mixers.qatten import QattenMixer
from envs.matrix_game import print_matrix_status
from utils.rl_utils import build_td_lambda_targets, build_q_lambda_targets
import torch as th
from torch.optim import RMSprop, Adam
import numpy as np
from utils.th_utils import get_parameters_num
from utils.attack_util import logits_margin, attack, get_diff, noise_atk, get_max_diff
from torch.autograd import Variable
from utils.pareto import MinNormSolver, gradient_normalizers
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
        mean_confidence_Q = (chosen_action_qvals * mask).sum() / mask.sum()

        return mean_confidence_Q


    def calculate_adversarial_loss(self, batch: EpisodeBatch, mac_index: int):
        """
        Calculate robustness metrics for a specific MAC in the population.
        Exactly follows the robustness computation logic in train() function.
        
        Args:
            batch: EpisodeBatch containing experience data
            mac_index: Index of the MAC in self.population to evaluate
            
        Returns:
            adv_loss: Mean adversarial loss for the specified MAC (scalar tensor)
        """
        # Get the Genome from population
        pop_genome = self.population[mac_index]
        
        # Get the relevant quantities for robust regularizer (same as in train())
        adv_terminated = batch["terminated"].float()
        adv_mask = batch["filled"].float()
        adv_mask[:, 1:] = adv_mask[:, 1:] * (1 - adv_terminated[:, :-1])
        
        # Set both agents to eval mode for evaluation
        pop_genome.mac1.agent.eval()
        pop_genome.mac2.agent.eval()
        
        adv_margin = []
        pop_genome.init_hidden(batch.batch_size)
        
        with th.no_grad():  # No gradients needed for evaluation
            for t in range(batch.max_seq_length):
                # Normal Q-values from Genome (min of two MACs)
                agent_outs = pop_genome.forward(batch, t=t)
                
                # Calculate adversarial margin
                adv_batch = noise_atk(batch, self.args)
                adv_agent_out = pop_genome.forward(adv_batch, t=t)
                adv_margin.append(get_max_diff(agent_outs, adv_agent_out))
        
        # Calculate adversarial loss (EXACTLY as in train())
        adv_margin = th.stack(adv_margin, dim=0).transpose(0, 1) * adv_mask.squeeze(2)  # (batch, seq_len)
        adv_loss = adv_margin.sum() / adv_mask.sum()
        
        return adv_loss
    
    
    def Evolve_pop(self):
        """
        Evolution operations for the population.
        TODO: Implement evolution strategy (selection, crossover, mutation)
        """
        pass
        
    def train(self, batch: EpisodeBatch, t_env: int, episode_num: int, per_weight=None):
        # Get the relevant quantities
        rewards = batch["reward"][:, :-1] # the total reward
        actions = batch["actions"][:, :-1]
        terminated = batch["terminated"][:, :-1].float()
        mask = batch["filled"][:, :-1].float()
        mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1]) # whether the sample is valid
        avail_actions = batch["avail_actions"]
        
        # just for robust regularizer
        adv_terminated = batch["terminated"].float()
        adv_mask = batch["filled"].float()
        adv_mask[:, 1:] = adv_mask[:, 1:] * (1 - adv_terminated[:, :-1])
        
        
        # Calculate estimated Q-Values for BOTH MACs in Genome
        self.Genome.train()  # Set both mac1 and mac2 to train mode
        
        # Forward pass for mac1
        mac_out_1 = []
        adv_margin = []
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
            
        # Calculate adversarial margin using min(q1, q2)
        if self.args.diff_regular or self.args.pareto:
            self.Genome.init_hidden(batch.batch_size)
            for t in range(batch.max_seq_length):
                agent_outs = self.Genome.forward(batch, t=t)  # min(q1, q2)
                adv_batch = noise_atk(batch, self.args)
                adv_agent_out = self.Genome.forward(adv_batch, t=t)    
                adv_margin.append(get_max_diff(agent_outs, adv_agent_out))
                
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
        

    # if t_env >= self.args.t_max // 4:        
        
        if self.args.robust_regular or self.args.diff_regular or self.args.pareto: 
            
            adv_margin = th.stack(adv_margin, dim=0).transpose(0, 1) * adv_mask.squeeze(2) # (batch, seq_len)
            adv_loss = adv_margin.sum() / adv_mask.sum()
            if self.args.weight_td:
                loss = th.exp(-self.log_var_a) * loss + self.log_var_a
                loss += (adv_loss * th.exp(-self.log_var_b) + self.log_var_b)
            elif self.args.weight_adv_loss:
                loss = loss + adv_loss * (-self.log_var_b) + (float(self.args.robust_lambda) / self.log_var_b)
            elif self.args.pareto:
                
                # lossa = masked_td_error
                # lossb = adv_margin[:, :-1].unsqueeze(2)
                # if th.matmul(lossa.transpose(1,2), lossb).sum() >= th.matmul(lossa.transpose(1,2), lossa).sum():
                #     gamma = 1
                # elif th.matmul(lossa.transpose(1,2), lossb).sum() >= th.matmul(lossb.transpose(1,2), lossb).sum():
                #     gamma = 0
                # else:
                #     gamma = th.matmul((lossb - lossa).transpose(1,2), lossb).mean() / th.norm(lossa - lossb, p=2) ** 2
                # lossa = lossa.sum() / mask.sum()
                # lossb = lossb.sum() / adv_mask.sum()
                # loss = gamma * lossa + (1 - gamma) * lossb
                
                # compute grad of td
                loss_data = {}
                grads = {}
                scale = {}
                tasks = ['td', 'adv']
                loss_data['td'] = loss.data
                loss_data['adv'] = adv_loss.data
                self.optimiser.zero_grad()
                loss.backward(retain_graph=True)
                grads['td'] = []
                for p in self.params:
                    if p.grad is not None:
                        grads['td'].append(Variable(p.grad.data.clone(), requires_grad=False))
                    
                # compute grad of adv    
                self.optimiser.zero_grad()
                adv_loss.backward()
                grads['adv'] = []
                for p in self.params:
                    if p.grad is not None:
                        grads['adv'].append(Variable(p.grad.data.clone(), requires_grad=False))
                        
                        
                # normalize
                gn = gradient_normalizers(grads, loss_data, self.args.normalization_type)
                for t in tasks:
                    for gr_i in range(len(grads[t])):
                        grads[t][gr_i] = grads[t][gr_i] / gn[t]
                        
                # use F-W algorithm to compute pareto optimal
                sol, _ = MinNormSolver.find_min_norm_element([grads[t] for t in tasks])
                for i, t in enumerate(tasks):
                    scale[t] = float(sol[i])
                    
                    
                self.optimiser.zero_grad()
                
                td_loss, adv_loss = self.compute_td_and_adv(batch, t_env, episode_num, per_weight) 
                
                # scale['adv'] = min(0.01, scale['adv'])
                # scale['td'] = 1 - scale['adv']
                                    
                loss = scale['td'] * td_loss + scale['adv'] * adv_loss
                        
            else: # fixed lambda
                loss += adv_loss * float(self.args.robust_lambda)
            
                

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

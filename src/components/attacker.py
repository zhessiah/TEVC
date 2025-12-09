import numpy as np
import torch as th
from torch.optim import RMSprop
import torch.nn.functional as F
import torch.nn as nn
from tqdm import tqdm
import os
from components.action_selectors import REGISTRY as action_REGISTRY
from components.episode_buffer import ReplayBuffer
import copy

class MLPAttacker(nn.Module):
    def __init__(self, args, load=False):
        super(MLPAttacker, self).__init__()
        self.args = args
        if load==True:
            self.p_ref =  [args.load_sparse_ref_delta/args.n_agents for _ in range(args.n_agents)]+[1-args.load_sparse_ref_delta]
        # set reference distribution
        else:   # sparse_ref_delta: shared attack prob
            if self.args.sparse_ref_delta == 0:
                self.p_ref = [1/(self.args.n_agents+1) for _ in range(args.n_agents+1)]
            else:
                self.p_ref =  [args.sparse_ref_delta/args.n_agents for _ in range(args.n_agents)]+[1-args.sparse_ref_delta]
        self.p_ref = th.FloatTensor(self.p_ref).to(self.args.device)
        self.lamb = args.spare_lambda
        
        input_shape = args.state_shape
        if args.concat_left_time:
            input_shape += 1
        self.fc1 = nn.Linear(input_shape, args.attacker_hidden_dim)
        self.fc2 = nn.Linear(args.attacker_hidden_dim, args.attacker_hidden_dim)
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_agents+1) # decide which agent to attack
        
        # Initialize optimizer first (before creating target_net to avoid copying it)
        self.optimiser = RMSprop(params=self.parameters(), lr=args.attack_lr, 
                                alpha=args.optim_alpha, eps=args.optim_eps)
        
        # Initialize target network (for Soft Q-Learning)
        # Use underscore prefix to prevent PyTorch from registering it as a submodule
        # This avoids the circular reference issue with state_dict()
        self._target_net = copy.deepcopy(self)
        
        # Freeze target network - no gradients needed
        for param in self._target_net.parameters():
            param.requires_grad = False
        
        # Buffer will be set externally via setup_buffer()
        self.buffer = None
        self.soft_tau = args.attacker_soft_tau
        self.to(self.args.device)
    
    @property
    def target_net(self):
        """Access target network (stored as _target_net to avoid PyTorch submodule registration)."""
        return self._target_net

    def forward(self, inputs):
        q = F.relu(self.fc1(inputs))
        q = F.relu(self.fc2(q))
        q = self.fc3(q)
        return q

    def batch_forward(self, ep_batch, t):
        bs = ep_batch.batch_size
        inputs = []
        inputs.append(ep_batch["state"][:, t])  # bs, state_shape
        if self.args.concat_left_time:
            inputs.append(ep_batch["left_attack"][:, t])
        inputs = th.cat([x.reshape(bs, -1) for x in inputs], dim=1)
        assert inputs.device == next(self.parameters()).device
        # get outputs
        attacker_outs = self.forward(inputs)  # bs, n_agents+1
        """if th.any(th.isnan(attacker_outs)):
            print(inputs)
            print(attacker_outs)
            assert 0"""
        return attacker_outs
    
    def setup_buffer(self, scheme, groups, preprocess):
        """Setup replay buffer for this attacker."""
        self.buffer = ReplayBuffer(scheme, groups, self.args.attacker_buffer_size,
                                   self.args.episode_limit+1, preprocess=preprocess,
                                   device="cpu" if self.args.buffer_cpu_only else self.args.device)
    
    def soft_update_target(self):
        """Soft update of target network: target = (1-tau)*target + tau*online"""
        for param, target_param in zip(self.parameters(), self.target_net.parameters()):
            target_param.data.copy_((1-self.soft_tau)*target_param.data + self.soft_tau*param.data)
    
    def store(self, episode_batch):
        """Store episode batch in replay buffer."""
        if self.buffer is None:
            raise RuntimeError("Buffer not initialized. Call setup_buffer() first.")
        self.buffer.insert_episode_batch(episode_batch)
    
    # def train(self, logger=None, log_step=None):
    #     """
    #     Train the attacker using Soft Q-Learning.
        
    #     Args:
    #         logger: Optional logger for logging training statistics
    #         log_step: Optional step number for logging
            
    #     Returns:
    #         bool: True if training succeeded, False if failed (e.g., NaN gradients)
    #     """
    #     # Check if buffer is ready
    #     if self.buffer is None:
    #         raise RuntimeError("Buffer not initialized. Call setup_buffer() first.")
        
    #     if not self.buffer.can_sample(self.args.attack_batch_size):
    #         return True  # Not an error, just not enough samples yet
        
    #     # Sample batch
    #     batch = self.buffer.sample(self.args.attack_batch_size)
    #     max_ep_t = batch.max_t_filled()
    #     batch = batch[:, :max_ep_t]
        
    #     # Move to device if needed
    #     if batch.device != self.args.device:
    #         batch.to(self.args.device)
        
    #     # Extract components
    #     rewards = batch["reward"][:, :-1]
    #     if self.args.shaping_reward:
    #         rewards = batch["shaping_reward"][:, :-1]
    #     actions = batch["action"][:, :-1]  # batch_size, max_seq_length-1, 1
    #     terminated = batch["terminated"][:, :-1].float()
    #     mask = batch["terminated"][:, :-1].float()
    #     mask[:, 1:] = mask[:, 1:] * (1 - terminated[:, :-1])
        
    #     # Compute Q-values for current states
    #     attacker_qs = []
    #     for t in range(batch.max_seq_length):
    #         attacker_q = self.batch_forward(batch, t=t)
    #         attacker_qs.append(attacker_q)
    #     attacker_qs = th.stack(attacker_qs, dim=1)  # batch_size, max_seq_length, ac_dim~n_agent+1
    #     chosen_action_qvals = th.gather(attacker_qs[:, :-1], dim=-1, index=actions)
    #     # (batch_size, max_seq_length-1, 1)
        
    #     # Compute target Q-values using target network
    #     targeted_attacker_qs = []
    #     for t in range(batch.max_seq_length):
    #         targeted_attacker_q = self.target_net.batch_forward(batch, t=t)
    #         targeted_attacker_qs.append(targeted_attacker_q)
    #     targeted_attacker_qs = th.stack(targeted_attacker_qs[1:], dim=1)
    #     # batch_size, max_seq_length-1, ac_dim~n_agent+1
        
    #     # Soft Q-Learning target: y = r + gamma * lambda * log(E_pref(a')[exp(Q(s',a')/lambda)])
    #     lamb = self.lamb
    #     targeted_attacker_q = lamb * th.log(
    #         (th.exp(targeted_attacker_qs/lamb) * self.p_ref).sum(dim=2)
    #     ).unsqueeze(2)
    #     targets = rewards + self.args.gamma * (1-terminated) * targeted_attacker_q
        
    #     # Compute TD-error and loss
    #     td_error = (chosen_action_qvals - targets.detach())
    #     mask = mask.expand_as(td_error)
    #     masked_td_error = td_error * mask
    #     loss = (masked_td_error ** 2).sum() / mask.sum()
        
    #     # Optimize
    #     self.optimiser.zero_grad()
    #     loss.backward()
    #     grad_norm = th.nn.utils.clip_grad_norm_(self.parameters(), self.args.grad_norm_clip)
        
    #     # Check for NaN gradients
    #     if th.any(th.isnan(grad_norm)):
    #         print(f"Warning: NaN gradients detected in attacker training!")
    #         return False
        
    #     self.optimiser.step()
    #     self.soft_update_target()
        
    #     # Log statistics if logger provided
    #     if logger is not None and log_step is not None:
    #         logger.log_stat("attacker_quality_loss", loss.item(), log_step)
    #         logger.log_stat("attacker_grad_norm", grad_norm.item(), log_step)
        
    #     return True


class Population:
    """
    Deprecated: This class is kept for backward compatibility only.
    Use MLPAttacker directly instead.
    
    Population manages multiple MLPAttacker instances for backward compatibility.
    For new code, create MLPAttacker instances directly and call their train() method.
    """
    def __init__(self, args):
        self.args = args
        self.size = args.pop_size
        self.attack_num = args.attack_num
        self.episode_limit = self.args.individual_sample_episode
        self.attack_agent_selector = action_REGISTRY[args.attack_agent_selector](args)
        self.attackers = []
        self.logger = None  # Will be set externally if needed

    def generate_attackers(self):
        """Generate a population of attackers."""
        candidates = []
        for _ in range(self.size):
            candidates.append(MLPAttacker(self.args))
        return candidates

    def reset(self, attackers):
        """Reset population with given attackers."""
        self.attackers = attackers
        assert len(self.attackers) == self.size, print(len(self.attackers), self.size)

    def setup_buffer(self, scheme, groups, preprocess):
        """Setup buffer for each attacker in the population."""
        if self.args.one_buffer:
            # Create one shared buffer
            self.buffer = ReplayBuffer(scheme, groups, self.args.attacker_buffer_size,
                                      self.args.episode_limit+1, preprocess=preprocess,
                                      device="cpu" if self.args.buffer_cpu_only else self.args.device)
            # Share the buffer with all attackers
            for attacker in self.attackers:
                attacker.buffer = self.buffer
        else:
            # Each attacker gets its own buffer
            for attacker in self.attackers:
                attacker.setup_buffer(scheme, groups, preprocess)

    def get_behavior_info(self, mac, runner):
        """Evaluate behavior of all attackers in the population."""
        last_attack_points = [[] for _ in range(self.size)]
        last_returns = [[] for _ in range(self.size)]
        last_won = [[] for _ in range(self.size)]
        for i, attacker in enumerate(self.attackers):
            mac.set_attacker(attacker)
            runner.setup_mac(mac)
            for k in range(self.args.attacker_eval_num):
                _, episode_batch, mixed_points, attack_cnt, epi_return, won = runner.run(test_mode=True)
                if k < 6:
                    last_attack_points[i] += mixed_points[:max(1, attack_cnt)]
                # need to be -return!!!
                last_returns[i].append(-epi_return)
                last_won[i].append(won)
        last_mean_return = [np.mean(x) for x in last_returns]
        last_won = [np.mean(x) for x in last_won]
        return last_attack_points, last_mean_return, last_won

    def store(self, episode_batch, mixed_points, attack_cnt, attacker_id):
        """Store episode data for a specific attacker."""
        # Store in the attacker's buffer
        self.attackers[attacker_id].store(episode_batch)

    def train(self, gen, train_step):
        """
        Train all attackers in the population.
        
        Note: This simplified version trains each attacker independently.
        Diversity loss has been removed. For diversity-aware training,
        implement it externally using the individual attackers.
        """
        success = True
        for i, attacker in enumerate(self.attackers):
            # Train this attacker
            log_this_step = (train_step == self.args.population_train_steps)
            train_success = attacker.train(
                logger=self.logger if log_this_step else None,
                log_step=gen if log_this_step else None
            )
            if not train_success:
                success = False
        
        return success

    def cuda(self):
        """Move all attackers to CUDA."""
        for attacker in self.attackers:
            attacker.cuda()
            if hasattr(attacker, 'target_net'):
                attacker.target_net.cuda()

    def save_models(self, path):
        """Save all attacker models."""
        for i in range(len(self.attackers)):
            th.save(self.attackers[i].state_dict(), "{0}/attacker_{1}.th".format(path, i))

    def load_models(self, load_path):
        """Load attacker models from disk."""
        attackers = []
        for i in range(len(os.listdir(load_path))):
            full_name = os.path.join(load_path, f"attacker_{i}.th")
            attacker = MLPAttacker(self.args, load=True).to(self.args.device)
            attacker.load_state_dict(th.load(full_name, map_location=lambda storage, loc: storage))
            attackers.append(attacker)
        self.reset(attackers)
    
    def long_eval(self, mac, runner, logger, threshold=0.8, num_eval=100, save_path=None):
        """Perform long evaluation of all attackers."""
        logger.console_logger.info(f"Start long eval, with {len(self.attackers)} attacker(s)")
        all_returns = []
        all_wons = []
        for attacker_id, attacker in enumerate(self.attackers):
            mac.set_attacker(attacker)
            runner.setup_mac(mac)
            returns = []
            wons = []
            for _ in tqdm(range(num_eval)):
                _, episode_batch, mixed_points, attack_cnt, epi_return, won = runner.run(test_mode=True)
                returns.append(-epi_return)
                wons.append(won)
            all_returns.append(np.mean(returns))
            all_wons.append(np.mean(wons))
            print("this attacker", attacker_id, " long eval returns: ", all_returns[-1])
            print("this attacker", attacker_id, " long eval won rate: ", all_wons[-1])
        print(
            f"mean of test {len(all_returns)} attackers: return: {np.mean(all_returns)}, win_rate: {np.mean(all_wons)}")


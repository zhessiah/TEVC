from envs import REGISTRY as env_REGISTRY
from functools import partial
from components.episode_buffer import EpisodeBatch
from multiprocessing import Pipe, Process
import numpy as np
import torch as th
import copy
from components.attacker import MLPAttacker

# Based (very) heavily on SubprocVecEnv from OpenAI Baselines
# https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
class Robust_ParallelRunner:

    def __init__(self, args, logger):
        self.args = args
        self.logger = logger
        self.batch_size = self.args.batch_size_run        

        # Make subprocesses for the envs
        self.parent_conns, self.worker_conns = zip(*[Pipe() for _ in range(self.batch_size)])
        env_fn = env_REGISTRY[self.args.env]
        self.ps = []
        for i, worker_conn in enumerate(self.worker_conns):
            ps = Process(target=env_worker, 
                    args=(worker_conn, CloudpickleWrapper(partial(env_fn, **self.args.env_args))))
            self.ps.append(ps)

        for p in self.ps:
            p.daemon = True
            p.start()

        self.parent_conns[0].send(("get_env_info", None))
        self.env_info = self.parent_conns[0].recv()
        self.episode_limit = self.env_info["episode_limit"]

        self.t = 0

        self.t_env = 0

        self.train_returns = []
        self.test_returns = []
        self.test_under_attack_returns = []  # 新增：攻击下的测试returns
        self.train_stats = {}
        self.test_stats = {}
        self.test_under_attack_stats = {}  # 新增：攻击下的测试stats

        self.log_train_stats_t = -100000
        self.attacker_pop_size = args.attacker_pop_size
        
        env_info = self.get_env_info()
        self.args.n_agents = env_info["n_agents"]
        self.args.n_actions = env_info["n_actions"]
        self.args.state_shape = env_info["state_shape"]
        self.args.episode_limit = env_info["episode_limit"]

        
        self.attacker_population = []
        for i in range(self.attacker_pop_size):
            self.attacker_population.append(MLPAttacker(args))
        
        # Initialize global episode counter for evaluate_mode (persistent across multiple run_under_attack calls)
        self.global_evaluate_episode_counter = 0


    def setup(self, scheme, groups, preprocess, mac):
        self.new_batch = partial(EpisodeBatch, scheme, groups, self.batch_size, self.episode_limit + 1,
                                 preprocess=preprocess, device=self.args.device)
        self.mac = mac
        self.scheme = scheme
        self.groups = groups
        self.preprocess = preprocess

    def get_env_info(self):
        return self.env_info

    def save_replay(self):
        pass

    def close_env(self):
        for parent_conn in self.parent_conns:
            parent_conn.send(("close", None))

    def reset(self):
        self.batch = self.new_batch()

        # Reset the envs
        for parent_conn in self.parent_conns:
            parent_conn.send(("reset", None))
            

        pre_transition_data = {
            "state": [],
            "avail_actions": [],
            "obs": [],
            "left_attack": []
        }
        # Get the obs, state and avail_actions back
        for parent_conn in self.parent_conns:
            data = parent_conn.recv()
            pre_transition_data["state"].append(data["state"]) # global state
            pre_transition_data["avail_actions"].append(data["avail_actions"])
            pre_transition_data["obs"].append(data["obs"])
            pre_transition_data["left_attack"].append([np.array([1.0])])

        self.batch.update(pre_transition_data, ts=0)

        self.t = 0
        self.env_steps_this_run = 0
    

    def run(self, mac, test_mode=False):
        self.mac = mac
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []  # may store extra stats like battle won. this is filled in ORDER OF TERMINATION
        attack_cnts = [0 for _ in range(self.batch_size)]
        
        if test_mode==True:
            attack_num = self.args.num_attack_test
        else:
            attack_num = self.args.num_attack_train
        
        # do_attack_num = 0
        # attack_num_agents = self.args.attack_num_agents
        # initial_attack_flag = copy.deepcopy(self.args.attack_duration)
        # save_probs = getattr(self.args, "save_probs", False)
        while True:
            ori_actions, byzantine_actions, victim_id = self.mac.select_byzantine_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            
            # Update the actions taken
            actions_chosen = {
                "actions": ori_actions.unsqueeze(1).to("cpu"), # original actions, not attacked
                "byzantine_actions": byzantine_actions.unsqueeze(1).to("cpu"), # attacked actions (all)
                "victim_id": victim_id.to("cpu")
            }

            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)



            # begin replacing actions for attacked agents
            do_actions = copy.deepcopy(ori_actions)
            
            # can attack
            actual_batch_idx = 0
            for idx in envs_not_terminated:
                if attack_cnts[idx] < attack_num: # can attack
                    attack_agent_id = victim_id[actual_batch_idx].data
                    if attack_agent_id != self.args.n_agents: # choose agent to attack
                        attack_cnts[idx] += 1
                        do_actions[actual_batch_idx][attack_agent_id] = copy.deepcopy(byzantine_actions[actual_batch_idx][attack_agent_id])
                        actual_batch_idx += 1
            do_actions = do_actions.to("cpu").numpy()
            # Send actions to each env
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated: # We produced actions for this env
                    if not terminated[idx]: # Only send the actions to the env if it hasn't terminated
                        parent_conn.send(("step", do_actions[action_idx]))
                    action_idx += 1 # actions is not a list over every env

            # Update envs_not_terminated
            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            # Post step data we will insert for the current timestep
            post_transition_data = {
                # "actions": ori_actions.to("cpu").numpy(),
                # "do_actions": do_actions,
                "reward": [],
                "terminated": []
            }
            # Data for the next step we will insert in order to select an action
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "left_attack": []
            }

            # Receive data back for each unterminated env
            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    # Remaining data for this current timestep
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # Data for the next timestep needed to select an action
                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    if attack_num:
                        pre_transition_data["left_attack"].append([(1 - attack_cnts[idx] / attack_num,)])
                    else:
                        pre_transition_data["left_attack"].append([0])
            # Add post_transiton data into the batch
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            # Move onto the next timestep
            self.t += 1

            # Add the pre-transition data
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        # Get stats back for each env
        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # 使用标准的统计容器（原始逻辑）
        cur_stats = self.test_stats if test_mode else self.train_stats
        cur_returns = self.test_returns if test_mode else self.train_returns
        log_prefix = "test_" if test_mode else ""
        
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch

    
    def run_under_attack(self, mac, test_mode=False, evaluate_mode=False):
        """
        运行episode进行训练或测试（攻击版本 - 用于测试对抗鲁棒性）
        
        与 run() 逻辑完全相同，但使用独立的统计容器：
        - test_under_attack_stats
        - test_under_attack_returns
        - log_prefix = "test_under_attack_"
        
        用于对抗训练的双测试模式。
        
        当 evaluate_mode=True 时，记录每个episode内部的详细数据：
        - 每步的奖励
        - 每个agent的局部Q值
        - 系统整体Q值
        - 攻击者的剩余预算
        - 攻击者选择攻击哪个智能体
        """
        self.mac = mac
        self.reset()

        all_terminated = False
        episode_returns = [0 for _ in range(self.batch_size)]
        episode_lengths = [0 for _ in range(self.batch_size)]
        self.mac.init_hidden(batch_size=self.batch_size)
        terminated = [False for _ in range(self.batch_size)]
        envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
        final_env_infos = []
        attack_cnts = [0 for _ in range(self.batch_size)]
        
        # === Evaluate Mode: Initialize data collection for each episode ===
        if evaluate_mode:
            import os
            from tensorboard_logger import Logger as TBLogger
            
            # Use the current experiment directory (with timestamp) as base
            # This is where the main training logs are stored
            if hasattr(self.args, 'model_path') and self.args.model_path:
                base_log_dir = os.path.join(self.args.model_path, "evaluate_episodes")
            else:
                # Fallback to local_results_path if model_path not available
                base_log_dir = os.path.join(self.args.local_results_path, "tb_logs", "evaluate_episodes")
            
            os.makedirs(base_log_dir, exist_ok=True)
            
            # Track within-episode data for each parallel environment
            episode_data = []
            cumulative_rewards = []  # Track cumulative reward for each environment
            episode_step_counters = []  # Track step within current episode for each environment
            episode_tb_loggers = []  # TensorBoard Logger instance for each environment's current episode
            
            for _ in range(self.batch_size):
                episode_data.append({
                    'rewards': [],         # Immediate reward at each step
                    'agent_qs': [],        # List of [q_agent0, q_agent1, ...] per step
                    'global_qs': [],       # Global Q value (sum of agent Qs)
                    'victim_ids': [],      # Which agent is attacked (or n_agents if none)
                    'left_attacks': []     # Remaining attack budget (normalized 0-1)
                })
                cumulative_rewards.append(0.0)  # Initialize cumulative reward to 0
                episode_step_counters.append(0)  # Initialize episode step counter
                episode_tb_loggers.append(None)  # Initialize logger placeholder
            
            # NOTE: Do NOT reset episode_counters here - use global persistent counter
            # This ensures each episode gets a unique ID across multiple run_under_attack() calls
            
            global_step_counter = 0  # Global step counter across all episodes and environments
        
        if test_mode==True:
            attack_num = self.args.num_attack_test
        else:
            attack_num = self.args.num_attack_train
        
        while True:
            ori_actions, byzantine_actions, victim_id = self.mac.select_byzantine_actions(self.batch, t_ep=self.t, t_env=self.t_env, bs=envs_not_terminated, test_mode=test_mode)
            
            # Get Q values if in evaluate mode
            selected_qs = None
            global_qs = None
            if evaluate_mode:
                try:
                    # Call forward with correct parameter name 't' instead of 't_ep'
                    forward_output = self.mac.forward(self.batch, t=self.t, test_mode=test_mode)
                    if isinstance(forward_output, tuple):
                        agent_outs, qs = forward_output
                    else:
                        qs = forward_output
                    # qs shape: [batch_size, n_agents, n_actions]
                    selected_qs = []
                    for b_idx in range(len(envs_not_terminated)):
                        agent_qs = []
                        for a_idx in range(self.args.n_agents):
                            action = ori_actions[b_idx, a_idx].item()
                            if action < qs.shape[-1]:
                                agent_qs.append(qs[b_idx, a_idx, action].item())
                            else:
                                agent_qs.append(0.0)
                        selected_qs.append(agent_qs)
                    global_qs = [sum(sq) for sq in selected_qs]
                except Exception as e:
                    self.logger.console_logger.warning(f"[Evaluate Mode] Failed to get Q values: {e}")
                    selected_qs = None
                    global_qs = None
            
            actions_chosen = {
                "actions": ori_actions.unsqueeze(1).to("cpu"),
                "byzantine_actions": byzantine_actions.unsqueeze(1).to("cpu"),
                "victim_id": victim_id.to("cpu")
            }

            self.batch.update(actions_chosen, bs=envs_not_terminated, ts=self.t, mark_filled=False)

            do_actions = copy.deepcopy(ori_actions)
            
            actual_batch_idx = 0
            for idx in envs_not_terminated:
                if attack_cnts[idx] < attack_num:
                    attack_agent_id = victim_id[actual_batch_idx].data
                    if attack_agent_id != self.args.n_agents:
                        attack_cnts[idx] += 1
                        do_actions[actual_batch_idx][attack_agent_id] = copy.deepcopy(byzantine_actions[actual_batch_idx][attack_agent_id])
                        actual_batch_idx += 1
            do_actions = do_actions.to("cpu").numpy()
            
            action_idx = 0
            for idx, parent_conn in enumerate(self.parent_conns):
                if idx in envs_not_terminated:
                    if not terminated[idx]:
                        parent_conn.send(("step", do_actions[action_idx]))
                    action_idx += 1

            envs_not_terminated = [b_idx for b_idx, termed in enumerate(terminated) if not termed]
            all_terminated = all(terminated)
            if all_terminated:
                break

            post_transition_data = {
                "reward": [],
                "terminated": []
            }
            pre_transition_data = {
                "state": [],
                "avail_actions": [],
                "obs": [],
                "left_attack": []
            }

            for idx, parent_conn in enumerate(self.parent_conns):
                if not terminated[idx]:
                    data = parent_conn.recv()
                    post_transition_data["reward"].append((data["reward"],))

                    episode_returns[idx] += data["reward"]
                    episode_lengths[idx] += 1
                    if not test_mode:
                        self.env_steps_this_run += 1

                    env_terminated = False
                    if data["terminated"]:
                        final_env_infos.append(data["info"])
                    if data["terminated"] and not data["info"].get("episode_limit", False):
                        env_terminated = True
                    terminated[idx] = data["terminated"]
                    post_transition_data["terminated"].append((env_terminated,))

                    # === Evaluate Mode: Record within-episode data ===
                    # Record data from ALL environments (each gets its own episode folder)
                    if evaluate_mode and idx in envs_not_terminated:
                        batch_idx = envs_not_terminated.index(idx)
                        
                        # Create new TensorBoard logger for this episode if not exists
                        if episode_tb_loggers[idx] is None:
                            # Create unique directory for this episode using global counter
                            episode_log_dir = os.path.join(base_log_dir, f"episode_{self.global_evaluate_episode_counter}")
                            # Create an independent Logger instance for this episode
                            episode_tb_loggers[idx] = TBLogger(episode_log_dir)
                            self.logger.console_logger.info(
                                f"[Evaluate Mode] Starting Episode {self.global_evaluate_episode_counter} - "
                                f"Logging to {episode_log_dir}"
                            )
                            # Increment counter immediately after creating logger to ensure unique episode IDs
                            # for all parallel environments (critical for batch_size > 1)
                            self.global_evaluate_episode_counter += 1
                        
                        # Update cumulative reward for this environment
                        cumulative_rewards[idx] += data["reward"]
                        
                        # Get Q values for this environment
                        selected_q = selected_qs[batch_idx] if selected_qs is not None else [0.0] * self.args.n_agents
                        global_q = global_qs[batch_idx] if global_qs is not None else 0.0
                        
                        # Get victim ID (which agent is attacked)
                        victim = victim_id[batch_idx].item()
                        
                        # Calculate remaining attack budget
                        left_attack = 1 - attack_cnts[idx] / attack_num if attack_num else 0.0
                        
                        # Log data to episode-specific TensorBoard
                        # Use episode_step_counters[idx] as x-axis (step within episode)
                        episode_tb_loggers[idx].log_value("cumulative_reward", cumulative_rewards[idx], episode_step_counters[idx])
                        episode_tb_loggers[idx].log_value("global_q", global_q, episode_step_counters[idx])
                        
                        # Log individual agent Q values
                        for agent_id, agent_q in enumerate(selected_q):
                            episode_tb_loggers[idx].log_value(f"agent{agent_id}_q", agent_q, episode_step_counters[idx])
                        
                        # Log victim ID (which agent is being attacked)
                        episode_tb_loggers[idx].log_value("victim_id", victim, episode_step_counters[idx])
                        
                        # Log remaining attack budget
                        episode_tb_loggers[idx].log_value("attack_budget", left_attack, episode_step_counters[idx])
                        
                        # Increment counters
                        global_step_counter += 1
                        episode_step_counters[idx] += 1
                        
                        # Store data for episode summary logging
                        episode_data[idx]['rewards'].append(data["reward"])
                        episode_data[idx]['agent_qs'].append(selected_q)
                        episode_data[idx]['global_qs'].append(global_q)
                        episode_data[idx]['victim_ids'].append(victim)
                        episode_data[idx]['left_attacks'].append(left_attack)
                        
                        # If episode just terminated, reset logger and counters
                        if data["terminated"]:
                            num_steps = len(episode_data[idx]['rewards'])
                            total_reward = sum(episode_data[idx]['rewards'])
                            avg_q = sum(episode_data[idx]['global_qs']) / num_steps if num_steps > 0 else 0.0
                            
                            # Calculate average reward per step
                            avg_reward = total_reward / num_steps if num_steps > 0 else 0.0
                            
                            # Get battle result (win=1, loss=0) from environment info
                            battle_won = data["info"].get("battle_won", 0)
                            win_value = 1 if battle_won else 0
                            
                            # Log battle result and average reward as scalar values (logged at final step)
                            # This creates single values per episode
                            episode_tb_loggers[idx].log_value("battle_won", win_value, episode_step_counters[idx] - 1)
                            episode_tb_loggers[idx].log_value("average_reward", avg_reward, episode_step_counters[idx] - 1)
                            
                            # The episode number is global_counter - 1 (since we already incremented when creating)
                            completed_episode_num = self.global_evaluate_episode_counter - 1
                            win_status = "WON" if win_value == 1 else "LOST"
                            self.logger.console_logger.info(
                                f"[Evaluate Mode] Episode {completed_episode_num} completed ({win_status}): "
                                f"{num_steps} steps, total_reward={total_reward:.2f}, avg_reward={avg_reward:.4f}, avg_q={avg_q:.2f}"
                            )
                            
                            # Reset logger for next episode (tensorboard-logger doesn't need explicit close)
                            episode_tb_loggers[idx] = None
                            
                            # Reset data for this environment (for next episode)
                            episode_data[idx] = {
                                'rewards': [],
                                'agent_qs': [],
                                'global_qs': [],
                                'victim_ids': [],
                                'left_attacks': []
                            }
                            cumulative_rewards[idx] = 0.0  # Reset cumulative reward for next episode
                            episode_step_counters[idx] = 0  # Reset episode step counter
                            # Note: Do NOT increment counter here - already incremented when creating logger

                    pre_transition_data["state"].append(data["state"])
                    pre_transition_data["avail_actions"].append(data["avail_actions"])
                    pre_transition_data["obs"].append(data["obs"])
                    if attack_num:
                        pre_transition_data["left_attack"].append([(1 - attack_cnts[idx] / attack_num,)])
                    else:
                        pre_transition_data["left_attack"].append([0])
            
            self.batch.update(post_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=False)
            self.t += 1
            self.batch.update(pre_transition_data, bs=envs_not_terminated, ts=self.t, mark_filled=True)

        if not test_mode:
            self.t_env += self.env_steps_this_run

        for parent_conn in self.parent_conns:
            parent_conn.send(("get_stats",None))

        env_stats = []
        for parent_conn in self.parent_conns:
            env_stat = parent_conn.recv()
            env_stats.append(env_stat)

        # 使用攻击测试的统计容器
        cur_stats = self.test_under_attack_stats if test_mode else self.train_stats
        cur_returns = self.test_under_attack_returns if test_mode else self.train_returns
        log_prefix = "test_under_attack_" if test_mode else ""
        
        infos = [cur_stats] + final_env_infos

        cur_stats.update({k: sum(d.get(k, 0) for d in infos) for k in set.union(*[set(d) for d in infos])})
        cur_stats["n_episodes"] = self.batch_size + cur_stats.get("n_episodes", 0)
        cur_stats["ep_length"] = sum(episode_lengths) + cur_stats.get("ep_length", 0)

        cur_returns.extend(episode_returns)

        n_test_runs = max(1, self.args.test_nepisode // self.batch_size) * self.batch_size
        if test_mode and (len(self.test_under_attack_returns) == n_test_runs):
            self._log(cur_returns, cur_stats, log_prefix)
        elif self.t_env - self.log_train_stats_t >= self.args.runner_log_interval:
            self._log(cur_returns, cur_stats, log_prefix)
            if hasattr(self.mac.action_selector, "epsilon"):
                self.logger.log_stat("epsilon", self.mac.action_selector.epsilon, self.t_env)
            self.log_train_stats_t = self.t_env

        return self.batch
    
    
    def _log(self, returns, stats, prefix):
        self.logger.log_stat(prefix + "return_mean", np.mean(returns), self.t_env)
        self.logger.log_stat(prefix + "return_std", np.std(returns), self.t_env)
        returns.clear()

        for k, v in stats.items():
            if k != "n_episodes":
                self.logger.log_stat(prefix + k + "_mean" , v/stats["n_episodes"], self.t_env)
        stats.clear()


def env_worker( remote, env_fn): # core of the python multiprocess
# Make environment
    env = env_fn.x()
    while True:
        cmd, data = remote.recv()
        if cmd == "step":
            actions = data
            # Take a step in the environment
            reward, terminated, env_info = env.step(actions)
            # Return the observations, avail_actions and state to make the next action
            state = env.get_state()
            avail_actions = env.get_avail_actions()
            obs = env.get_obs()
            remote.send({
                # Data for the next timestep needed to pick an action
                "state": state,
                "avail_actions": avail_actions,
                "obs": obs,
                # Rest of the data for the current timestep
                "reward": reward,
                "terminated": terminated,
                "info": env_info
            })
        elif cmd == "reset":
            env.reset() 
            remote.send({
                "state": env.get_state(),
                "avail_actions": env.get_avail_actions(),
                "obs": env.get_obs()
            })
        elif cmd == "close":
            env.close()
            remote.close()
            break
        elif cmd == "get_env_info":
            remote.send(env.get_env_info())
        elif cmd == "get_stats":
            remote.send(env.get_stats())
        elif cmd == "perturb_on":
            env.perturb_on()
        elif cmd == "perturb_off":
            env.perturb_off()
        else:
            raise NotImplementedError


class CloudpickleWrapper():
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries to use pickle)
    """
    def __init__(self, x):
        self.x = x
    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)
    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


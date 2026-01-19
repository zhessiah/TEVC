import datetime
import os
import pprint
import time
import threading
import torch as th
from types import SimpleNamespace as SN
from utils.logging import Logger
from utils.timehelper import time_left, time_str
from os.path import dirname, abspath
import numpy as np


from learners import REGISTRY as le_REGISTRY
from runners import REGISTRY as r_REGISTRY
from controllers import REGISTRY as mac_REGISTRY
from components.episode_buffer import ReplayBuffer
from components.transforms import OneHot
from components.Evolver import NN_Evolver, Genome
from components.attacker import MLPAttacker

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits


def rl_to_evo(rl_genome, evo_genome):
    """
    Transfer parameters from RL Genome to Evolution Genome with HARD REPLACEMENT.
    Both Genomes contain two MACs (mac1 and mac2).
    Direction: RL → Evolution (完全替换，不保留原参数)
    
    NOTE: Mixer is NOT transferred (it's not part of Genome anymore).
    NOTE: This is the legacy single-individual version. 
    For batch replacement, use rl_to_evo_excluding_pareto() instead.
    """
    # HARD REPLACEMENT: Direct parameter copy (no EMA)
    # Transfer mac1 parameters
    for target_param, param in zip(evo_genome.mac1.agent.parameters(), rl_genome.mac1.agent.parameters()):
        target_param.data.copy_(param.data)  # Complete replacement
    
    # Transfer mac2 parameters
    for target_param, param in zip(evo_genome.mac2.agent.parameters(), rl_genome.mac2.agent.parameters()):
        target_param.data.copy_(param.data)  # Complete replacement


def rl_to_evo_excluding_elites(rl_genome, population, best_agents):
    """
    Transfer RL Genome parameters to ALL population members EXCEPT Pareto first front.
    This is a HARD REPLACEMENT - completely overwrites inferior individuals with RL parameters.
    
    Args:
        rl_genome: Main RL Genome (source)
        population: List of evolution Genomes (targets)
    
    Direction: RL → Evolution (每次训练后完全替换，不保留原参数)
    Strategy: Complete parameter copy (no EMA, full replacement)
    
    NOTE: Only MACs are transferred. Mixer stays in learner.
    """
    for i in range(len(population)):
        # Skip Pareto first front individuals (elites)
        if i in best_agents:
            continue
        
        evo_genome = population[i]
        
        # HARD REPLACEMENT: Directly copy RL parameters (no EMA, no blending)
        # Transfer mac1 parameters
        for target_param, param in zip(evo_genome.mac1.agent.parameters(), rl_genome.mac1.agent.parameters()):
            target_param.data.copy_(param.data)  # Complete replacement
        
        # Transfer mac2 parameters
        for target_param, param in zip(evo_genome.mac2.agent.parameters(), rl_genome.mac2.agent.parameters()):
            target_param.data.copy_(param.data)  # Complete replacement


def run(_run, _config, _log):

    # check args sanity
    _config = args_sanity_check(_config, _log)

    args = SN(**_config)
    args.device = "cuda" if args.use_cuda else "cpu"

    # setup loggers
    logger = Logger(_log)

    _log.info("Experiment Parameters:")
    experiment_params = pprint.pformat(_config,
                                       indent=4,
                                       width=1)
    _log.info("\n\n" + experiment_params + "\n")

    # configure tensorboard logger
    unique_token = "{}__{}".format(args.name, datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S"))
    args.unique_token = unique_token
    if args.use_tensorboard:
        tb_logs_direc = os.path.join(dirname(dirname(dirname(abspath(__file__)))), "results", "tb_logs")
        
        if args.EA:
            tb_logs_direc = os.path.join(tb_logs_direc, "EA")
        if args.adversarial_training:
            tb_logs_direc = os.path.join(tb_logs_direc, "adversarial_training")
        # elif args.test_attack:
        #     tb_logs_direc = os.path.join(tb_logs_direc, "test_attack")
        # elif args.all_test_attack:
        #     tb_logs_direc = os.path.join(tb_logs_direc, "all_test_attack")
        # else:
        #     tb_logs_direc = os.path.join(tb_logs_direc, "no_attack")
        
        
        if "sc2" in args.env or "gfootball" in args.env:
            tb_logs_direc = os.path.join(tb_logs_direc, args.env_args["map_name"])
        elif "mpe" in args.env:
            tb_logs_direc = os.path.join(tb_logs_direc, args.env_args["key"], str(args.env_args['num_agents']) + '_agents')
        elif "stag_hunt" in args.env:
            tb_logs_direc = os.path.join(tb_logs_direc, args.env, str(args.env_args['n_agents']) + '_agents')
        else:
            tb_logs_direc = os.path.join(tb_logs_direc, args.env)
            
            
        # tb_logs_direc = os.path.join(tb_logs_direc, "epsilon_{}".format(args.epsilon))
        
        # if args.pareto:
        #     tb_logs_direc = os.path.join(tb_logs_direc, "pareto")
        # elif args.robust_regular:
        #     tb_logs_direc = os.path.join(tb_logs_direc, "robust_regular")
        #     tb_logs_direc = os.path.join(tb_logs_direc, str(args.robust_lambda))
        # elif args.diff_regular:
        #     tb_logs_direc = os.path.join(tb_logs_direc, "diff_regular")
        #     tb_logs_direc = os.path.join(tb_logs_direc, str(args.robust_lambda))
        tb_exp_direc = os.path.join(tb_logs_direc, "{}").format(unique_token)
        logger.setup_tb(tb_exp_direc)
        args.model_path = tb_exp_direc


    # sacred is on by default
    logger.setup_sacred(_run)

    # Run and train
    run_sequential(args=args, logger=logger)

    # Clean up after finishing
    print("Exiting Main")

    print("Stopping all threads")
    for t in threading.enumerate():
        if t.name != "MainThread":
            print("Thread {} is alive! Is daemon: {}".format(t.name, t.daemon))
            t.join(timeout=1)
            print("Thread joined")

    print("Exiting script")

    # Making sure framework really exits
    os._exit(os.EX_OK)


def evaluate_sequential(mac, args, runner):

    for _ in range(args.test_nepisode):
        _, _ = runner.run(mac, test_mode=True)  # Unpack tuple, ignore attacker_stats

    if args.save_replay:
        runner.save_replay()

    runner.close_env()

def run_sequential(args, logger):

    # Init runner so we can get env info
    runner = r_REGISTRY[args.runner](args=args, logger=logger)

    # Set up schemes and groups here
    env_info = runner.get_env_info()
    args.n_agents = env_info["n_agents"]
    args.n_actions = env_info["n_actions"]
    args.state_shape = env_info["state_shape"]
    args.accumulated_episodes = getattr(args, "accumulated_episodes", None)

    # if getattr(args, 'agent_own_state_size', False):
    #     args.agent_own_state_size = get_agent_own_state_size(args.env_args)

    # Default/Base scheme
    scheme = {
        "state": {"vshape": env_info["state_shape"]},
        "obs": {"vshape": env_info["obs_shape"], "group": "agents"},
        "actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "do_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "byzantine_actions": {"vshape": (1,), "group": "agents", "dtype": th.long},
        "victim_id" : {"vshape": (1,), "dtype": th.float32},
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
        "left_attack": {"vshape": (1,), "dtype": th.float32}, 
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)]),
        "do_actions": ("do_actions_onehot", [OneHot(out_dim=args.n_actions)]),
    }    

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    
    evolver = NN_Evolver(args)
    
    # Create a separate evolver for attackers with same configuration
    attacker_evolver = NN_Evolver(args) if args.EA and args.adversarial_training else None

    # Setup populations of multiagent controller here
    if args.EA:
        # Create main Genome (contains two MACs for double Q-learning)
        genome = Genome(args, buffer, groups)
        
        # Create population of Genomes
        population = []
        # fitness = np.zeros((args.pop_size, 3))
        for i in range(args.pop_size):
            population.append(Genome(args, buffer, groups))
            
        population_attackers = []
        for i in range(args.attacker_pop_size):
            population_attackers.append(MLPAttacker(args))
        pop_size = args.pop_size
        elite_size = args.elite_size 
        attacker_pop_size = args.attacker_pop_size
         
        best_agents = list(range(int(pop_size*elite_size)))
        best_attackers = list(range(int(attacker_pop_size*elite_size)))
    
    else:
        mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        population = []
    
    
    
    
    

    # Give runner the scheme
    if args.EA:
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=genome)
    else:
        runner.setup(scheme=scheme, groups=groups, preprocess=preprocess, mac=mac)

    # Learner
    if args.EA:
        learner = le_REGISTRY[args.learner](genome, population, buffer.scheme, logger, args)
    else:
        learner = le_REGISTRY[args.learner](mac, buffer.scheme, logger, args)
    

    if args.use_cuda:
        learner.cuda()

    if args.checkpoint_path != "":

        timesteps = []
        timestep_to_load = 0

        if not os.path.isdir(args.checkpoint_path):
            logger.console_logger.info("Checkpoint directiory {} doesn't exist".format(args.checkpoint_path))
            return
        
        # Go through all files in args.checkpoint_path
        for name in os.listdir(args.checkpoint_path):
            full_name = os.path.join(args.checkpoint_path, name)
            # Check if they are dirs the names of which are numbers
            if os.path.isdir(full_name) and name.isdigit():
                timesteps.append(int(name))

        if args.load_step == 0:
            # choose the max timestep
            timestep_to_load = max(timesteps)
        else:
            # choose the timestep closest to load_step
            timestep_to_load = min(timesteps, key=lambda x: abs(x - args.load_step))

        model_path = os.path.join(args.checkpoint_path, str(timestep_to_load))

        logger.console_logger.info("Loading model from {}".format(model_path))
        learner.load_models(model_path)
        runner.t_env = timestep_to_load

        if args.evaluate or args.save_replay:
            if args.EA:
                evaluate_sequential(genome, args, runner)
            else:
                evaluate_sequential(mac, args, runner)
            return

    # start training
    episode = 0
    last_test_T = -args.test_interval - 1
    last_log_T = 0
    model_save_time = 0

    start_time = time.time()
    last_time = start_time
    
    # Learning-assisted dynamic weighting mechanism
    TD_init = None  # Initial TD error (average of first ~1000 steps)
    TD_ema = None  # Exponential Moving Average of TD error
    ema_beta = 0.99  # EMA smoothing factor
    learning_progress = 0.0  # Current learning progress P_t ∈ [0, 1]
    alpha_t = 0.0  # Dynamic mixing coefficient for robustness vs optimality
    
    # Transfer function parameters for alpha_t
    P_center = getattr(args, 'alpha_center', 0.5)  # Center point for sigmoid
    sigmoid_k = getattr(args, 'alpha_steepness', 10.0)  # Steepness of sigmoid
    
    # Alternative: Linear ramp parameters
    use_linear_ramp = getattr(args, 'use_linear_alpha', False)
    P_start = getattr(args, 'alpha_start', 0.3)  # Start ramping up
    P_end = getattr(args, 'alpha_end', 0.8)  # Full robustness focus
    
    TD_history = []  # Store TD errors for initialization
    init_steps = getattr(args, 'td_init_steps', 50)  # Steps to compute TD_init
    
    logger.console_logger.info("Beginning training for {} timesteps".format(args.t_max))
    
    # Log global attack parameters (MACO-specific) to TensorBoard
    if args.use_tensorboard and hasattr(args, 'num_attack_train') and hasattr(args, 'num_attack_test'):
        logger.log_stat("config/num_attack_train", args.num_attack_train, runner.t_env)
        logger.log_stat("config/num_attack_test", args.num_attack_test, runner.t_env)

    while runner.t_env <= args.t_max:
        
        # only the best agents in the population interact with the env
        # run with bests of pop
        # fitness = [[] for _ in range(args.pop_size)]
        replace_index = None
        if args.EA and runner.t_env > args.start_timesteps:
            
            with th.no_grad():
                for i in best_agents:
                    if args.adversarial_training:
                        for j in best_attackers:
                            population[i].mac1.set_attacker(population_attackers[j])
                            episode_batch = runner.run(population[i], test_mode=False)
                            buffer.insert_episode_batch(episode_batch)
                    else:
                        episode_batch = runner.run(population[i], test_mode=False)
                        buffer.insert_episode_batch(episode_batch)
        #  normal run with mac
        # else:
        #     elite_index = [0]
        #     fitness = np.zeros((args.pop_size, 3))
        # Normal run with mac (or genome if EA)
        with th.no_grad():
            if args.EA:
                if args.adversarial_training:
                    for j in best_attackers:
                        genome.mac1.set_attacker(population_attackers[j])
                        episode_batch = runner.run(genome, test_mode=False)
                else:
                    episode_batch = runner.run(genome, test_mode=False)
            else:
                episode_batch = runner.run(mac, test_mode=False)
            buffer.insert_episode_batch(episode_batch)
            
            

        if buffer.can_sample(args.batch_size):
            # training start
            next_episode = episode + args.batch_size_run
            if args.accumulated_episodes and next_episode % args.accumulated_episodes != 0:
                continue

            episode_sample = buffer.sample(args.batch_size)

            # Truncate batch to only filled timesteps
            max_ep_t = episode_sample.max_t_filled()
            episode_sample = episode_sample[:, :max_ep_t]

            if episode_sample.device != args.device:
                episode_sample.to(args.device)
            # if (args.training_attack and runner.t_env >= args.t_max // 3 and runner.t_env <= args.t_max // 3 * 2):
            #     learner.args.robust_regular = True
            # else:
            #     learner.args.robust_regular = False
            info = learner.train(episode_sample, runner.t_env, episode)
            
            # ========== RL → Population Injection (每次训练后) ==========
            # After each training step, inject RL knowledge to all non-elite population members
            # This ensures inferior individuals benefit from RL's gradient-based learning
            if args.EA and runner.t_env > args.start_timesteps:
                # Determine which individuals to protect from replacement
                # Inject RL Genome to all population members EXCEPT best agents
                rl_to_evo_excluding_elites(genome, population, best_agents)
                

            # Extract TD error from training info (assuming learner returns a dict with TD info)
            # If learner.train() returns the info dict directly
            current_TD = info['td_error']
            # === Learning-Assisted Dynamic Weighting Mechanism ===
            # Step 1: Initialize TD_init from first 50 training steps (only once!)
            if TD_init is None:
                TD_history.append(current_TD)
                if len(TD_history) >= init_steps:
                    TD_init = np.mean(TD_history)
                    TD_ema = TD_init  # Initialize EMA with TD_init
                    logger.console_logger.info(f"TD_init computed: {TD_init:.6f} (baseline from first {init_steps} steps)")
            
            # Step 2: Update EMA of TD error (TD_init remains constant!)
            else:
                # TD_init is already set, just update the EMA
                TD_ema = ema_beta * TD_ema + (1 - ema_beta) * current_TD
                
                # Step 3: Compute Learning Progress P_t
                learning_progress = max(0.0, min(1.0, 1.0 - TD_ema / (TD_init + 1e-8)))
                
                # Step 4: Compute dynamic weight alpha_t
                if use_linear_ramp:
                    # Linear ramp transfer function
                    if learning_progress < P_start:
                        alpha_t = 0.0
                    elif learning_progress > P_end:
                        alpha_t = 1.0
                    else:
                        alpha_t = (learning_progress - P_start) / (P_end - P_start)
                else:
                    # Logistic sigmoid transfer function
                    alpha_t = 1.0 / (1.0 + np.exp(-sigmoid_k * (learning_progress - P_center)))
                
                # Log the adaptive weights
                if episode % 100 == 0:  # Log every 100 episodes
                    logger.console_logger.info(
                        f"[Learning-Assisted EA] Episode {episode} - "
                        f"Progress: {learning_progress:.3f}, Alpha_t: {alpha_t:.3f}, "
                        f"TD_ema: {TD_ema:.6f}, TD_init: {TD_init:.6f}"
                    )
                    if args.use_tensorboard:
                        logger.log_stat("learning_progress", learning_progress, runner.t_env)
                        logger.log_stat("alpha_t", alpha_t, runner.t_env)
                        logger.log_stat("td_ema", TD_ema, runner.t_env)
                        logger.log_stat("td_init", TD_init, runner.t_env)
            
            del episode_sample
            
            # ========== Start Evo ==========
            if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
                # print('EA starts')
                
                # === Step 1: DEPRECATED - Memetic SGD Injection ===
                # MACO_overview.md V6 refactoring: Explicit memetic learning is REMOVED
                # Main agent learning IS the implicit memetic mechanism
                # Only memetic injection (rl_to_evo_excluding_elites) is used
                use_memetic_sgd = False  # Force disable (deprecated)
                     
                # === Step 2: Fitness Evaluation (Simplified to 2 Objectives) ===
                # MACO V6: Defenders have only 2 objectives
                # 1. Optimality: TD Error (environment fitting)
                # 2. Robustness: Fault Isolation Ratio (Byzantine tolerance)
                fitness = [[] for _ in range(args.pop_size)]
                
                for i in range(args.pop_size):
                    # Objective 1: Optimality (TD Error)
                    env_precise_fitness = learner.calculate_TD_error(episode_batch, i)
                    fitness[i].append(-env_precise_fitness.cpu().numpy())
                    
                    # Objective 2: Robustness (Fault Isolation Ratio)
                    robust_smooth_fitness = learner.calculate_adversarial_loss(episode_batch, i)
                    fitness[i].append(-robust_smooth_fitness.cpu().numpy())
                
                # === Step 3: Evolutionary Selection (进化选择阶段) ===
                # Pass alpha_t to evolver for learning-assisted dynamic weighting
                elite_index, replace_index = evolver.epoch(population, fitness, agent_level=True, alpha_t=alpha_t)
                best_agents = elite_index
                
                # Log defender fitness statistics (all defenders)
                if args.use_tensorboard:
                    for i in range(args.pop_size):
                        # Objective 1: TD Error (optimality) - 注意 fitness 存的是负值
                        logger.log_stat(f"defender_{i}_td_error", -fitness[i][0], runner.t_env)
                        # Objective 2: Robustness (fault isolation)
                        logger.log_stat(f"defender_{i}_robustness", -fitness[i][1], runner.t_env)
                    
                    # Also log population statistics
                    td_errors = [-f[0] for f in fitness]
                    robustness_scores = [-f[1] for f in fitness]
                    logger.log_stat("defender_td_error_mean", np.mean(td_errors), runner.t_env)
                    logger.log_stat("defender_td_error_std", np.std(td_errors), runner.t_env)
                    logger.log_stat("defender_robustness_mean", np.mean(robustness_scores), runner.t_env)
                    logger.log_stat("defender_robustness_std", np.std(robustness_scores), runner.t_env)
                
                # Update elite archive with current elites
                # if use_adversarial_novelty and len(elite_index) > 0:
                #     learner.update_elite_archive(elite_index, episode_batch)
                
                logger.console_logger.info(
                    f"[EA Selection] New generation created. "
                    f"Elites: {elite_index[:5]}, Replace: {replace_index}"
                )
                
                # === Step 4: Closing the Memetic Loop (参数回注阶段) ===
                
                # 4a. RL → Evolution: 用 RL 智能体替换最差个体（注入快速学习能力）
                # NOTE: This step is now handled AFTER EACH TRAINING in the training loop above
                # We keep this for backward compatibility but it's redundant with the new logic
                use_rl_injection = getattr(args, 'use_rl_injection', True)  # Disable old logic by default
                if use_rl_injection and replace_index is not None:
                    for i in replace_index:
                        rl_to_evo(genome, population[i])
                        logger.console_logger.info(f"[RL → Evo (Legacy)] Injected RL Genome into population[{i}]")
                    evolver.rl_policy = replace_index 
                
                # ========== Attacker Evolution (Co-evolution) ==========
                if args.adversarial_training:
                    logger.console_logger.info("[Attacker Evolution] Starting attacker population evolution...")
                    
                    # === Step 1: Memetic SGD for Elite Attackers (NEW: Budget-Modulated Advantage) ===
                    # Uses new attack advantage function from MACO V6
                    use_attacker_memetic = getattr(args, 'use_attacker_memetic_sgd', True)
                    num_finetune_attackers = getattr(args, 'num_finetune_attackers', 3)
                    attacker_sgd_steps = getattr(args, 'attacker_memetic_sgd_steps', 1)
                    
                    if use_attacker_memetic:
                        logger.console_logger.info(
                            f"[Attacker Memetic SGD] Finetuning Top-{num_finetune_attackers} "
                            f"elite attackers with {attacker_sgd_steps} SGD steps (Attack Advantage)..."
                        )
                        
                        for attacker_idx in best_attackers[:num_finetune_attackers]:
                            # 微调该attacker使用Attack Advantage (budget-modulated)
                            quality_loss = learner.memetic_finetune_attacker(
                                population_attackers[attacker_idx], 
                                episode_batch, 
                                attacker_sgd_steps
                            )
                            
                            if args.use_tensorboard and attacker_idx == best_attackers[0]:
                                logger.log_stat("attacker_memetic_advantage_loss", quality_loss, runner.t_env)
                        
                        logger.console_logger.info(
                            f"[Attacker Memetic SGD] Completed. "
                            f"Elite {best_attackers[0]} Advantage Loss: {quality_loss:.6f}"
                        )
                    
                    # === Step 2: Calculate attacker fitness (Simplified to 2 Objectives) ===
                    # CORRECTED V6: TD Error Maximization with Counterfactual Actions
                    # 
                    # Mathematical Foundation:
                    # F_att,1(φ) = E_τ [(y - Q_tot(s, a_φ))²]
                    # 
                    # Where a_φ = counterfactual actions under attacker φ's Byzantine interference
                    # 
                    # Byzantine Attack Model:
                    # - Attacker φ selects victims via φ.batch_forward(batch, t)
                    # - Victim's actions are replaced with Byzantine (worst) actions
                    # - Different attackers → different victim selections → different actions → different TD errors
                    # 
                    # Key Efficiency:
                    # 1. NO environment rollout needed (offline batch computation)
                    # 2. Reuses batch's pre-computed byzantine_actions
                    # 3. Only recomputes victim selection (cheap forward pass)
                    # 4. Dense signal: per-timestep TD error (vs per-episode reward)
                    #
                    # Why this works:
                    # - calculate_attacker_TD_error() constructs counterfactual actions
                    # - Uses attacker i to select victims for each timestep
                    # - Replaces victim actions with byzantine_actions from batch
                    # - Computes Q(s, a_counterfactual) and TD error
                    # - Different attackers produce different counterfactual actions → different TD errors
                    attacker_fitness = [[] for _ in range(args.attacker_pop_size)]
                    
                    # === Step 2.1: Calculate novelty for ALL attackers at once (O(N) optimization) ===
                    # REFACTORED: Avoid redundant computation (was O(N²), now O(N))
                    with th.no_grad():
                        all_novelties = learner.calculate_all_attacker_novelties(
                            episode_batch, population_attackers
                        )
                    
                    # === Step 2.2: Calculate fitness for each attacker ===
                    for i in range(args.attacker_pop_size):
                        # Objective 1: TD Error Maximization (CORRECTED V6)
                        # Uses counterfactual action construction (no env rollout!)
                        with th.no_grad():
                            td_error = learner.calculate_attacker_TD_error(
                                episode_batch, i, population_attackers
                            )
                        
                        # Higher TD error = better attacker (more cognitive disruption)
                        attacker_fitness[i].append(td_error.item())
                        
                        # Objective 2: Behavioral Novelty (pre-computed above)
                        # Diversity to avoid local optima
                        attacker_fitness[i].append(all_novelties[i])
                    
                    # === Step 3: Evolutionary selection for attackers ===
                    attacker_elite_index, attacker_replace_index = attacker_evolver.epoch(
                        population_attackers, attacker_fitness, agent_level=True, alpha_t=alpha_t
                    )
                    best_attackers = attacker_elite_index
                    
                    logger.console_logger.info(
                        f"[Attacker Evolution] New attacker generation. "
                        f"Elites: {attacker_elite_index[:5]}"
                    )
                    
                    # Log attacker fitness statistics (all attackers)
                    if args.use_tensorboard:
                        for i in range(args.attacker_pop_size):
                            # Objective 1: TD Error (cognitive disruption metric)
                            logger.log_stat(f"attacker_{i}_td_error", attacker_fitness[i][0], runner.t_env)
                            # Objective 2: Behavioral Novelty (diversity metric)
                            logger.log_stat(f"attacker_{i}_novelty", attacker_fitness[i][1], runner.t_env)
                        
                        # Also log population statistics
                        td_errors = [f[0] for f in attacker_fitness]
                        novelties = [f[1] for f in attacker_fitness]
                        logger.log_stat("attacker_td_error_mean", np.mean(td_errors), runner.t_env)
                        logger.log_stat("attacker_td_error_std", np.std(td_errors), runner.t_env)
                        logger.log_stat("attacker_novelty_mean", np.mean(novelties), runner.t_env)
                        logger.log_stat("attacker_novelty_std", np.std(novelties), runner.t_env) 
        
        
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            
            # === 双测试: 对抗训练模式 ===
            if args.EA and args.adversarial_training:
                # 测试 1: 干净性能 (无攻击) - 使用 run()
                logger.console_logger.info("[测试1/2] 评估干净性能 (无攻击)...")
                original_num_attack_test = args.num_attack_test
                args.num_attack_test = 0
                
                for _ in range(n_test_runs):
                    runner.run(genome, test_mode=True)  
                
                args.num_attack_test = original_num_attack_test
                
                # 测试 2: 精英攻击者下的鲁棒性 - 使用 run_under_attack()
                if len(best_attackers) > 0:
                    logger.console_logger.info("[测试2/2] 评估对抗鲁棒性 (精英攻击者)...")
                    elite_attacker_id = best_attackers[0]
                    genome.mac1.set_attacker(population_attackers[elite_attacker_id])
                    
                    for _ in range(n_test_runs):
                        runner.run_under_attack(genome, test_mode=True)  # 使用 run_under_attack() -> test_under_attack_*
                    
                    logger.console_logger.info(f"[双测试完成] 精英攻击者 {elite_attacker_id}")
            else:
                # 标准测试 (单一测试)
                for _ in range(n_test_runs):
                    if args.EA:
                        runner.run(genome, test_mode=True)
                    else:
                        runner.run(mac, test_mode=True)

        


        if args.save_model and (runner.t_env - model_save_time >= args.save_model_interval or model_save_time == 0):
            model_save_time = runner.t_env
            save_path = os.path.join(args.model_path, "models", str(runner.t_env))
            #"results/models/{}".format(unique_token)
            os.makedirs(save_path, exist_ok=True)
            logger.console_logger.info("Saving models to {}".format(save_path))

            # learner should handle saving/loading -- delegate actor save/load to mac,
            # use appropriate filenames to do critics, optimizer states
            learner.save_models(save_path)

        episode += args.batch_size_run

        if (runner.t_env - last_log_T) >= args.log_interval:
            logger.log_stat("episode", episode, runner.t_env)
            logger.print_recent_stats()
            last_log_T = runner.t_env

    runner.close_env()
    logger.console_logger.info("Finished Training")


def args_sanity_check(config, _log):

    # set CUDA flags
    # config["use_cuda"] = True # Use cuda whenever possible!
    if config["use_cuda"] and not th.cuda.is_available():
        config["use_cuda"] = False
        _log.warning("CUDA flag use_cuda was switched OFF automatically because no CUDA devices are available!")

    if config["test_nepisode"] < config["batch_size_run"]:
        config["test_nepisode"] = config["batch_size_run"]
    else:
        config["test_nepisode"] = (config["test_nepisode"]//config["batch_size_run"]) * config["batch_size_run"]

    return config

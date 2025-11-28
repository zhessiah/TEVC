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

from smac.env import StarCraft2Env

def get_agent_own_state_size(env_args):
    sc_env = StarCraft2Env(**env_args)
    # qatten parameter setting (only use in qatten)
    return  4 + sc_env.shield_bits_ally + sc_env.unit_type_bits


def rl_to_evo(rl_genome, evo_genome):
    """
    Transfer parameters from RL Genome to Evolution Genome with exponential moving average.
    Both Genomes contain two MACs (mac1 and mac2).
    """
    alpha = 0.1
    # Transfer mac1 parameters
    for target_param, param in zip(evo_genome.mac1.agent.parameters(), rl_genome.mac1.agent.parameters()):
        target_param.data.copy_(alpha * target_param.data + (1 - alpha) * param.data)
    
    # Transfer mac2 parameters
    for target_param, param in zip(evo_genome.mac2.agent.parameters(), rl_genome.mac2.agent.parameters()):
        target_param.data.copy_(alpha * target_param.data + (1 - alpha) * param.data)

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
        if args.training_attack:
            tb_logs_direc = os.path.join(tb_logs_direc, "train_attack")
        elif args.test_attack:
            tb_logs_direc = os.path.join(tb_logs_direc, "test_attack")
        elif args.all_test_attack:
            tb_logs_direc = os.path.join(tb_logs_direc, "all_test_attack")
        else:
            tb_logs_direc = os.path.join(tb_logs_direc, "no_attack")
        
        
        if "sc2" in args.env or "gfootball" in args.env:
            tb_logs_direc = os.path.join(tb_logs_direc, args.env_args["map_name"])
        elif "mpe" in args.env:
            tb_logs_direc = os.path.join(tb_logs_direc, args.env_args["key"], str(args.env_args['num_agents']) + '_agents')
        elif "stag_hunt" in args.env:
            tb_logs_direc = os.path.join(tb_logs_direc, args.env, str(args.env_args['n_agents']) + '_agents')
        else:
            tb_logs_direc = os.path.join(tb_logs_direc, args.env)
            
            
        tb_logs_direc = os.path.join(tb_logs_direc, "epsilon_{}".format(args.epsilon))
        
        if args.pareto:
            tb_logs_direc = os.path.join(tb_logs_direc, "pareto")
        elif args.robust_regular:
            tb_logs_direc = os.path.join(tb_logs_direc, "robust_regular")
            tb_logs_direc = os.path.join(tb_logs_direc, str(args.robust_lambda))
        elif args.diff_regular:
            tb_logs_direc = os.path.join(tb_logs_direc, "diff_regular")
            tb_logs_direc = os.path.join(tb_logs_direc, str(args.robust_lambda))
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
        runner.run(mac, test_mode=True)

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
        "avail_actions": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.int},
        "probs": {"vshape": (env_info["n_actions"],), "group": "agents", "dtype": th.float},
        "reward": {"vshape": (1,)},
        "terminated": {"vshape": (1,), "dtype": th.uint8},
    }
    groups = {
        "agents": args.n_agents
    }
    preprocess = {
        "actions": ("actions_onehot", [OneHot(out_dim=args.n_actions)])
    }

    buffer = ReplayBuffer(scheme, groups, args.buffer_size, env_info["episode_limit"] + 1,
                          preprocess=preprocess,
                          device="cpu" if args.buffer_cpu_only else args.device)
    
    evolver = NN_Evolver(args)

    # Setup populations of multiagent controller here
    if args.EA:
        # Create main Genome (contains two MACs for double Q-learning)
        genome = Genome(args, buffer, groups)
        
        # Create population of Genomes
        population = []
        # fitness = np.zeros((args.pop_size, 3))
        for i in range(args.pop_size):
            population.append(Genome(args, buffer, groups))
        pop_size = args.pop_size
        elite_size = args.elite_size 
         
        best_agents = list(range(int(pop_size*elite_size)))
    
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

    while runner.t_env <= args.t_max:
        
        # only the best agents in the population interact with the env
        # run with bests of pop
        # fitness = [[] for _ in range(args.pop_size)]
        replace_index = None
        if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
            
            with th.no_grad():
                for i in best_agents:
                    episode_batch = runner.run(population[i], test_mode=False)
                    buffer.insert_episode_batch(episode_batch)
        #  normal run with mac
        # else:
        #     elite_index = [0]
        #     fitness = np.zeros((args.pop_size, 3))
        # Normal run with mac (or genome if EA)
        with th.no_grad():
            if args.EA:
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
            
            # NSGA begins here
            if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
                # print('EA starts')
                # Reset fitness to list of lists for append operations
                fitness = [[] for _ in range(args.pop_size)]
                
                # Determine which robustness confidence metric to use
                use_evolutionary_consensus = getattr(args, 'use_evolutionary_consensus', False)
                use_twin_confidence = getattr(args, 'use_twin_adversarial_confidence', False)
                use_adversarial_novelty = getattr(args, 'use_adversarial_novelty', False)
                
                # Get current elite indices for population consensus calculation
                # Use the best agents from previous epoch (if available)
                elite_indices = learner.best_agents if hasattr(learner, 'best_agents') else list(range(min(3, args.pop_size)))
                
                for i in range(args.pop_size):
                    # === Optimality Metrics (左翼) ===
                    # 1. Environment Fitness: 负 TD Error (衡量Q函数有多准)
                    env_precise_fitness = learner.calculate_TD_error(episode_batch, i)
                    fitness[i].append(-env_precise_fitness.cpu().numpy())
                    
                    # 2. Value Confidence: 期望 Q 值 (衡量策略有多强)
                    confidence_Q_fitness = learner.calculate_confidence_Q(episode_batch, i)
                    fitness[i].append(confidence_Q_fitness.cpu().numpy())
                    
                    # === Robustness Metrics (右翼) ===
                    # 3. Attack Sensitivity: 负 Global Q Smoothness (衡量受影响多小)
                    robust_smooth_fitness = learner.calculate_adversarial_loss(episode_batch, i)
                    fitness[i].append(-robust_smooth_fitness.cpu().numpy())
                    
                    # 4. Robustness Confidence / Evolutionary Consensus: 鲁棒性确定性或进化共识
                    if use_evolutionary_consensus:
                        # TEVC特供版：进化共识得分 (Evolutionary Consensus Score)
                        # 结合 Twin-Q 内部一致性 + 种群集成一致性
                        # F4 = -(KL(π_Q1||π_Q2) + β·KL(π_me||π_ensemble))
                        evolutionary_consensus = learner.calculate_evolutionary_consensus(
                            episode_batch, i, elite_indices
                        )
                        fitness[i].append(evolutionary_consensus.cpu().numpy())
                        
                    elif use_twin_confidence:
                        # 增强版：计算双重熵+分歧，取负得到"双重自信度"
                        adversarial_entropy = learner.calculate_twin_adversarial_entropy(episode_batch, i)
                        robustness_confidence = -adversarial_entropy  # 熵越低，自信度越高
                        fitness[i].append(robustness_confidence.cpu().numpy())
                    else:
                        # 基础版：计算单一熵，取负得到"自信度"
                        adversarial_entropy = learner.calculate_adversarial_entropy(episode_batch, i)
                        robustness_confidence = -adversarial_entropy  # 熵越低,自信度越高
                        fitness[i].append(robustness_confidence.cpu().numpy())
                    
                    # 5. Adversarial Behavioral Novelty (Quality-Diversity): 对抗性行为新颖度
                    if use_adversarial_novelty:
                        # TEVC特供版：基于存档的新颖度搜索 (Archive-Based Novelty Search)
                        # 衡量当前个体在对抗样本下的行为与历史精英的差异性
                        # F5 = avg_distance(π_me(·|o_adv), π_archive(·|o_adv))
                        adversarial_novelty = learner.calculate_adversarial_novelty(episode_batch, i)
                        fitness[i].append(adversarial_novelty.cpu().numpy())
                
                # Log EA execution with current alpha_t
                num_objectives = len(fitness[0])
                consensus_mode = "Evolutionary Consensus" if use_evolutionary_consensus else ("Twin-Q" if use_twin_confidence else "Basic")
                novelty_mode = " + Novelty" if use_adversarial_novelty else ""
                logger.console_logger.info(
                    f"[EA Epoch] Episode {episode}, Mode: {consensus_mode}{novelty_mode}, "
                    f"Objectives: {num_objectives}, Alpha_t: {alpha_t:.3f} "
                    f"(Optimality weight: {1-alpha_t:.3f}, Robustness weight: {alpha_t:.3f})"
                )
                
                # Pass alpha_t to evolver for learning-assisted dynamic weighting
                elite_index, replace_index = evolver.epoch(population, fitness, agent_level=True, alpha_t=alpha_t)
                best_agents = elite_index
                
                # Update elite archive with current elites
                if use_adversarial_novelty and len(elite_index) > 0:
                    learner.update_elite_archive(elite_index, episode_batch)
                
                # print("EA ends.")

        # Execute test runs once in a while

        if args.EA and runner.t_env > args.start_timesteps and episode % args.EA_freq == 0:
            # if args.Pareto:
            #     fitness = np.array(fitness)
            #     replace_index = np.argmin(fitness[:, 0]) # replace worst reward
            # else:
            #     replace_index = np.argmin(fitness)
            if replace_index is not None:
                for i in replace_index:
                    # prev_state = copy.deepcopy(population[replace_index].mac1.agent.state_dict())
                    rl_to_evo(genome, population[i]) # replace population[replace_index]'s params with genome's params
            # if evaluate_new_population(pop) < evaluate_previous_population(prev_state):
            #     pop[replace_index].mac1.agent.load_state_dict(prev_state)
            evolver.rl_policy = replace_index
            # print('Replace bad individuals with RL')
        
        
        n_test_runs = max(1, args.test_nepisode // runner.batch_size)
        if (runner.t_env - last_test_T) / args.test_interval >= 1.0:

            logger.console_logger.info("t_env: {} / {}".format(runner.t_env, args.t_max))
            logger.console_logger.info("Estimated time left: {}. Time passed: {}".format(
                time_left(last_time, last_test_T, runner.t_env, args.t_max), time_str(time.time() - start_time)))
            last_time = time.time()

            last_test_T = runner.t_env
            
            # if (args.test_attack and runner.t_env >= args.t_max // 3 and runner.t_env <= args.t_max // 3 * 2): # attack during the second third of the training
            if (args.all_test_attack):
                runner.set_perturb(True) 
            elif (args.test_attack and runner.t_env >= args.t_max // 3 and runner.t_env <= args.t_max // 3 * 2): # attack during the second third of the training
                runner.set_perturb(True)
            for _ in range(n_test_runs):
                if args.EA:
                    runner.run(genome, test_mode=True)
                else:
                    runner.run(mac, test_mode=True)
            runner.set_perturb(False)
        
            # Replace bad individuals with mac
        


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

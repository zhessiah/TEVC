import random
import numpy as np
import fastrand
import torch
from components.episode_buffer import EpisodeBatch
from controllers import REGISTRY as mac_REGISTRY

def is_lnorm_key(key):
    return key.startswith('lnorm')

class Genome:
    """
    Genome class containing a single MAC object.
    Simplified from Double Q-learning architecture as double-Q has minimal impact on results.
    """
    def __init__(self, args, buffer, groups):
        self.n_agents = args.n_agents
        self.args = args
        
        # Create a single MAC
        self.mac = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        
        # Use mac's action selector
        self.action_selector = self.mac.action_selector
        self.agent_output_type = args.agent_output_type
        self.save_probs = getattr(self.args, 'save_probs', False)
    
    def train(self):
        """Set agent to training mode."""
        self.mac.agent.train()
        
    def eval(self):
        """Set agent to evaluation mode."""
        self.mac.agent.eval()
    
    def init_hidden(self, batch_size):
        """Initialize hidden states for MAC."""
        self.mac.init_hidden(batch_size)
    
    def forward(self, ep_batch, t, test_mode=False):
        """
        Forward pass through MAC and return Q-values.
        Simplified from Double Q-learning as it has minimal impact on results.
        """
        return self.mac.forward(ep_batch, t, test_mode=test_mode)
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """Select actions using Q-values from MAC."""
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    
    def select_byzantine_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """Select byzantine actions using Q-values from MAC."""
        ori_actions, worst_actions, which_agent_to_attack = self.mac.select_actions(ep_batch, t_ep, t_env, bs, test_mode=test_mode)
        return ori_actions, worst_actions, which_agent_to_attack
    
    def parameters(self):
        """Return parameters from MAC."""
        return self.mac.parameters()

    def load_state(self, other_mac):
        """Load state from another Genome or MAC."""
        if hasattr(other_mac, 'mac'):
            # Loading from another Genome
            self.mac.load_state(other_mac.mac)
        else:
            # Loading from a single MAC
            self.mac.load_state(other_mac)

    def cuda(self):
        """Move MAC to CUDA."""
        self.mac.cuda()
    
    def save_models(self, path):
        """Save MAC model."""
        torch.save(self.mac.agent.state_dict(), "{}/agent.th".format(path))
    
    def load_models(self, path):
        """Load MAC model."""
        self.mac.agent.load_state_dict(
            torch.load("{}/agent.th".format(path), map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        """Build agents for MAC."""
        self.mac._build_agents(input_shape)

    def _build_inputs(self, batch, t):
        """Build inputs for agents (delegates to MAC's method)."""
        return self.mac._build_inputs(batch, t)

    def _get_input_shape(self, scheme):
        """Get input shape for agents (delegates to MAC's method)."""
        return self.mac._get_input_shape(scheme)
    
    def has_agent_level_support(self):
        """
        Check if MAC supports agent-level operations (MACO agent-level decomposition).
        
        Returns:
            bool: True if MAC has agent_W list (MACOAgentController), False otherwise
        """
        return hasattr(self.mac, 'agent_W') and isinstance(self.mac.agent_W, list)
    
    def get_agent_W(self, agent_index):
        """
        Get the weight head for a specific agent.
        
        Args:
            agent_index: Index of the agent (0 ~ n_agents-1)
            
        Returns:
            agent_W: Weight head network for the specified agent
            
        Raises:
            ValueError: If MAC does not support agent-level operations
        """
        if self.has_agent_level_support():
            return self.mac.agent_W[agent_index]
        else:
            raise ValueError("Current MAC does not support agent-level operations. "
                           "Use 'maco_agent_mac' controller for agent-level evolution.")
    
    def get_agent_SR(self):
        """
        Get the shared state representation network.
        
        Returns:
            agent_SR: Shared state representation network
            
        Raises:
            ValueError: If MAC does not support agent-level operations
        """
        if self.has_agent_level_support():
            return self.mac.agent_SR
        else:
            raise ValueError("Current MAC does not support agent-level operations. "
                           "Use 'maco_agent_mac' controller for agent-level evolution.")
    
    def state_dict(self):
        """
        Return state dict of MAC for evolutionary operations.
        Returns a flat dictionary of parameter tensors.
        
        Important: The returned tensors share storage with the actual parameters,
        so modifications to the returned tensors will affect the model parameters.
        This is required for the mutate_inplace() operation in NN_Evolver.
        
        Supports two MAC types:
        1. BasicMAC-style: has self.mac.agent (single shared agent)
        2. MACOAgentMAC-style: has self.mac.agent_SR + self.mac.agent_W (decomposed)
        """
        state = {}
        
        # Check which MAC type we have
        if hasattr(self.mac, 'agent_SR') and hasattr(self.mac, 'agent_W'):
            # MACOAgentMAC: decomposed architecture
            # Add agent_SR parameters
            for name, param in self.mac.agent_SR.named_parameters():
                state[f"agent_SR.{name}"] = param.data
            
            # Add agent_W parameters for each agent
            for i, agent_w in enumerate(self.mac.agent_W):
                for name, param in agent_w.named_parameters():
                    state[f"agent_W.{i}.{name}"] = param.data
        else:
            # BasicMAC: single shared agent
            for name, param in self.mac.agent.named_parameters():
                state[name] = param.data
        
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load state dict into MAC.
        
        Supports both BasicMAC and MACOAgentMAC architectures.
        """
        if hasattr(self.mac, 'agent_SR') and hasattr(self.mac, 'agent_W'):
            # MACOAgentMAC: need to split state_dict
            sr_state = {}
            w_states = [{}for _ in range(len(self.mac.agent_W))]
            
            for key, value in state_dict.items():
                if key.startswith("agent_SR."):
                    sr_state[key.replace("agent_SR.", "")] = value
                elif key.startswith("agent_W."):
                    # Parse agent_W.{i}.{param_name}
                    parts = key.split(".", 2)
                    agent_idx = int(parts[1])
                    param_name = parts[2]
                    w_states[agent_idx][param_name] = value
            
            self.mac.agent_SR.load_state_dict(sr_state)
            for i, agent_w in enumerate(self.mac.agent_W):
                agent_w.load_state_dict(w_states[i])
        else:
            # BasicMAC: direct load
            self.mac.agent.load_state_dict(state_dict)

class NN_Evolver:
    def __init__(self, args):
        self.current_gen = 0
        self.args = args
        self.prob_reset_and_sup = args.prob_reset_and_sup

        self.frac = args.frac
        self.population_size = self.args.pop_size
        self.num_elitists = int(self.args.elite_size * args.pop_size)
        if self.num_elitists < 1: self.num_elitists = 1

        self.rl_policy = None
        self.selection_stats = {'elite': 0, 'selected': 0, 'discarded': 0, 'total': 0.0000001}

    def selection_tournament(self, index_rank, num_offsprings, tournament_size):
        total_choices = len(index_rank)
        offsprings = []
        for i in range(num_offsprings):
            winner = np.min(np.random.randint(total_choices, size=tournament_size))
            offsprings.append(index_rank[winner])

        offsprings = list(set(offsprings))  # Find unique offsprings
        if len(offsprings) % 2 != 0:  # Number of offsprings should be even
            offsprings.append(offsprings[fastrand.pcg32bounded(len(offsprings))])
        return offsprings

    def list_argsort(self, seq):
        return sorted(range(len(seq)), key=seq.__getitem__)

    def regularize_weight(self, weight, mag):
        if weight > mag: weight = mag
        if weight < -mag: weight = -mag
        return weight

    def crossover_inplace(self, gene1, gene2):
        """
        FIXED: Correctly pair each weight matrix with its corresponding bias vector.
        
        Original Bug:
        - Used the LAST bias for ALL weight matrices
        - Caused IndexError when weight.shape[0] > last_bias.shape[0]
        
        Fix:
        - Look ahead to find the corresponding bias for each weight
        - Verify bias dimensions match: bias.shape[0] == weight.shape[0]
        - Only swap bias if it exists and dimensions match
        """
        # Convert parameters to list to enable lookahead
        params1 = list(gene1.parameters())
        params2 = list(gene2.parameters())
        
        i = 0
        while i < len(params1):
            param1 = params1[i]
            param2 = params2[i]
            
            W1 = param1.data
            W2 = param2.data
            
            # Only process weight matrices (2D tensors)
            if len(W1.shape) == 2:
                # Look ahead to find corresponding bias
                b_1 = None
                b_2 = None
                if i + 1 < len(params1):
                    next_param1 = params1[i + 1]
                    next_param2 = params2[i + 1]
                    
                    # Verify it's the corresponding bias: 1D and matching output dimension
                    if (len(next_param1.data.shape) == 1 and 
                        next_param1.data.shape[0] == W1.shape[0]):
                        b_1 = next_param1.data
                        b_2 = next_param2.data
                
                # Perform crossover on weight matrix
                num_variables = W1.shape[0]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)
                
                for _ in range(num_cross_overs):
                    receiver_choice = random.random()
                    ind_cr = fastrand.pcg32bounded(W1.shape[0])
                    
                    if receiver_choice < 0.5:
                        W1[ind_cr, :] = W2[ind_cr, :]
                        # Only swap bias if it exists and dimensions match
                        if b_1 is not None:
                            b_1[ind_cr] = b_2[ind_cr]
                    else:
                        W2[ind_cr, :] = W1[ind_cr, :]
                        # Only swap bias if it exists and dimensions match
                        if b_2 is not None:
                            b_2[ind_cr] = b_1[ind_cr]
            
            i += 1

    def mutate_inplace(self, gene, agent_level=False):
        """
        Mutate a genome in-place.
        
        Args:
            gene: Genome instance (defender) or MLPAttacker instance (attacker) to mutate
            agent_level: If True, perform agent-level mutation (only on agent_W networks)
                        Only applies to Genome with agent-level support (MACOMAC)
        
        Note: mutation_prob is already checked in epoch(), so we don't check it again here
        """
        # Check if gene supports agent-level operations (only Genome has this method)
        if agent_level and hasattr(gene, 'has_agent_level_support') and gene.has_agent_level_support():
            # Agent-level mutation: mutate each agent independently (MACOMAC defender)
            n_agents = len(gene.mac.agent_W)
            for agent_idx in range(n_agents):
                mutate_agent_level(gene, agent_idx, self.args)
            return
        
        # Parameter-level mutation (original implementation)
        trials = 5

        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10.0
        super_mut_prob = self.prob_reset_and_sup
        reset_prob = super_mut_prob + self.prob_reset_and_sup

        num_params = len(list(gene.parameters()))
        ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
        model_params = gene.state_dict()

        for i, key in enumerate(model_params):  # Mutate each param

            if is_lnorm_key(key):
                continue

            # References to the variable keys
            W = model_params[key]
            if len(W.shape) == 2:  # Weights, no bias
                ssne_prob = 1.0
                action_prob = ssne_probabilities[i]

                if random.random() < ssne_prob:
                    num_variables = W.shape[0]
                    # Crossover opertation [Indexed by row]
                    for index in range(num_variables):
                        random_num_num = random.random()
                        if random_num_num <= action_prob:
                            # print(W)
                            index_list = random.sample(range(W.shape[1]), int(W.shape[1] * self.frac))
                            random_num = random.random()
                            if random_num < super_mut_prob:  # Super Mutation probability
                                for ind in index_list:
                                    W[index, ind] += random.gauss(0, super_mut_strength * W[index, ind])
                            elif random_num < reset_prob:  # Reset probability
                                for ind in index_list:
                                    W[index, ind] = random.gauss(0, 1)
                            else:  # mutation even normal
                                for ind in index_list:
                                    W[index, ind] += random.gauss(0, mut_strength * W[index, ind])

                            # Regularization hard limit
                            W[index, :] = np.clip(W[index, :].cpu(), a_min=-1000000, a_max=1000000)

    def clone(self, master, replace):  # Replace the replace individual with master

        for target_param, source_param in zip(replace.parameters(), master.parameters()):
            target_param.data.copy_(source_param.data)
        # replacee.buffer.reset()
        # replacee.buffer.add_content_of(master.buffer)

    def reset_genome(self, gene):
        for param in (gene.parameters()):
            param.data.copy_(param.data)

    def pareto_dominates(self, individual1, individual2, alpha_t=0.0):
        """
        Weighted Pareto dominance based on learning progress.
        
        Symmetry Structure (4 objectives):
        - Optimality (左翼): 
            f[0] = -TD_error (环境拟合 - accuracy)
            f[1] = confidence_Q (价值自信 - ambition)
        - Robustness (右翼):
            f[2] = -adversarial_loss (攻击敏感度 - stability) 
            f[3] = adversarial_confidence (鲁棒自信度 - certainty)
        
        Extended Structure (5 objectives):
        - Optimality (左翼):
            f[0] = -TD_error (环境拟合 - accuracy)
            f[1] = confidence_Q (价值自信 - ambition)
        - Robustness + Diversity (右翼):
            f[2] = -adversarial_loss (攻击敏感度 - stability)
            f[3] = adversarial_confidence (鲁棒自信度 - certainty)
            f[4] = adversarial_novelty (对抗性新颖度 - diversity)
        
        Args:
            individual1, individual2: Fitness tuples (3D, 4D, or 5D)
            alpha_t: Learning progress weight ∈ [0, 1]
                - alpha_t = 0: Focus on optimality
                - alpha_t = 1: Focus on robustness
        
        Weighting scheme:
            w_optimality = 1 - alpha_t  (decreases as learning progresses)
            w_robustness = alpha_t      (increases as learning progresses)
        """
        # Determine number of objectives
        n_objectives = len(individual1)
        
        weights = [1.0 / n_objectives] * n_objectives
        
        # Scale each objective by its weight
        weighted_f1 = [f * w for f, w in zip(individual1, weights)]
        weighted_f2 = [f * w for f, w in zip(individual2, weights)]
        
        # Standard Pareto dominance on weighted objectives
        better_in_one = False
        for f1, f2 in zip(weighted_f1, weighted_f2):
            if f1 < f2:
                return False
            elif f1 > f2:
                better_in_one = True
        return better_in_one

    def non_dominated_sorting(self, population, alpha_t=0.0):
        pareto_fronts = []
        domination_counts = {i: 0 for i in range(len(population))} # 每个个体被支配的次数
        dominated_solutions = {i: [] for i in range(len(population))} # 每个个体支配的个体列表

        current_front = []

        for i in range(len(population)):
            for j in range(len(population)):
                if self.pareto_dominates(population[i], population[j], alpha_t=alpha_t):
                    dominated_solutions[i].append(j)
                elif self.pareto_dominates(population[j], population[i], alpha_t=alpha_t):
                    domination_counts[i] += 1

            if domination_counts[i] == 0:
                current_front.append(i)

        pareto_fronts.append(current_front)

        while len(current_front) > 0:
            next_front = []
            for i in current_front:
                for j in dominated_solutions[i]:
                    domination_counts[j] -= 1
                    if domination_counts[j] == 0:
                        next_front.append(j)
            pareto_fronts.append(next_front)
            current_front = next_front

        return pareto_fronts # 从前往后是优势度排序，同一个列表内的解是同一层级

    def fitness_split(self, fitness):
        """
        Split multi-objective fitness into individual components.
        
        Supports 3, 4, and 5 objective cases:
        - 3 objectives (legacy): TD_error, confidence_Q, adversarial_loss
        - 4 objectives: TD_error, confidence_Q, adversarial_loss, adversarial_confidence
        - 5 objectives: TD_error, confidence_Q, adversarial_loss, adversarial_confidence, adversarial_novelty
        """
        if len(fitness) == 5:
            # 5-objective case: Quality-Diversity structure
            env_precise_fitness = fitness[0]      # -TD_error (环境拟合)
            confidence_Q_fitness = fitness[1]     # confidence_Q (价值自信)
            adversarial_loss = fitness[2]         # -adversarial_loss (攻击敏感度)
            adversarial_confidence = fitness[3]   # adversarial_confidence (鲁棒自信度)
            adversarial_novelty = fitness[4]      # adversarial_novelty (对抗性新颖度)
            return env_precise_fitness, confidence_Q_fitness, adversarial_loss, adversarial_confidence, adversarial_novelty
        elif len(fitness) == 4:
            # 4-objective case: Symmetrical structure
            env_precise_fitness = fitness[0]      # -TD_error (环境拟合)
            confidence_Q_fitness = fitness[1]     # confidence_Q (价值自信)
            adversarial_loss = fitness[2]         # -adversarial_loss (攻击敏感度)
            adversarial_confidence = fitness[3]   # adversarial_confidence (鲁棒自信度)
            return env_precise_fitness, confidence_Q_fitness, adversarial_loss, adversarial_confidence
        elif len(fitness) == 3:
            # 3-objective case (backward compatibility)
            env_precise_fitness = fitness[0]
            confidence_Q_fitness = fitness[1]
            uncertainty_fitness = fitness[2]
            return env_precise_fitness, confidence_Q_fitness, uncertainty_fitness
        else:
            # Fallback: return as tuple
            return tuple(fitness)

    def calculate_crowding_distance(self, front, population):
        distances = {i: 0 for i in front}
        num_objectives = len(population[0])

        for m in range(num_objectives):
            sorted_front = sorted(front, key=lambda i: population[i][m])

            # # 给排序列表中的第一个和最后一个个体赋值无穷大，以防止选择时被过早丢弃
            # distances[sorted_front[0]] = distances[sorted_front[-1]] = float('inf')

            # 计算内部个体的拥挤距离（即相邻个体的目标值差异）
            for i in range(1, len(sorted_front) - 1):
                distances[sorted_front[i]] += (population[sorted_front[i + 1]][m] - population[sorted_front[i - 1]][m])

        return distances

    def epoch(self, pop, fitness_evals, agent_level=False, alpha_t=0.0, agent_importance=None):
        """
        One epoch of evolutionary algorithm with learning-assisted dynamic weighting.
        
        Args:
            pop: Population of Genome objects
            fitness_evals: List of fitness tuples for each individual
            agent_level: Whether to use agent-level operations
            alpha_t: Learning progress weight for adaptive multi-objective optimization
            agent_importance: numpy array of shape (n_agents,) indicating global importance (attack frequency)
                            If provided, Crossover prefers high-importance agents and Mutation targets low-importance ones.
        """
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)
        
        # Determine ranking strategy
        if self.args.Pareto:
            population = [self.fitness_split(fitness) for fitness in fitness_evals]
            pareto_fronts = self.non_dominated_sorting(population, alpha_t=alpha_t)
            
            # Calculate crowding distance for each front
            crowding_distances = {}
            for front in pareto_fronts:
                crowding_distances.update(self.calculate_crowding_distance(front, population))

            # Selection step: select individuals from Pareto fronts
            index_rank = []
            for front in pareto_fronts:
                index_rank.extend(front)
        else:
            index_rank = np.argsort(fitness_evals)[::-1]
        
        # Elitist indexes safeguard
        elitist_index = index_rank[:self.num_elitists]

        # ---------------------------------------------------------------------
        # CASE 1: AGENT-LEVEL EVOLUTION (Strict RACE Alignment)
        # ---------------------------------------------------------------------
        # If agent_level is True, proper RACE implementation requires independent 
        # evolution (Selection, Elitism, Crossover, Mutation) for each module.
        # This allows combinatorial mixing of best modules from different parents.
        if agent_level and hasattr(pop[0], 'has_agent_level_support') and pop[0].has_agent_level_support():
            # Define modules to evolve: -1 (SR) + 0..n_agents-1 (Heads)
            n_agents = len(pop[0].mac.agent_W)
            # We evolve SR first as an independent module (proxy for "Body")
            # Then evolve each Agent Head independently.
            modules_to_evolve = [-1] + list(range(n_agents))
            
            # Use a dummy list to track "Elites" for the return value (though elites differ per module)
            # We will use the elites from the LAST module (or SR) as the reference for "worst_index" calculation
            # But realistically, the concept of "Elite Individual" becomes "Composite Individual".
            final_elitists_indices = set() 
            
            # Calculate importance stats if available
            avg_importance = 0.5
            if agent_importance is not None:
                avg_importance = np.mean(agent_importance)

            for module_idx in modules_to_evolve:
                # 1. Selection Tournament (Per Module Selection)
                # If agent_importance is provided, adjust selection pressure?
                # Currently we stick to Fitness-based selection for parents for Stability.
                # But we adjust Mutation Pressure based on Importance.
                offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                                    tournament_size=3)
                
                # 2. Figure out unselected candidates (Per Module Logic)
                unselects = []
                new_elitists = [] # Track elites for THIS module to exclude them from mutation
                
                for i in range(self.population_size):
                    if i not in offsprings and i not in elitist_index:
                        unselects.append(i)
                random.shuffle(unselects)
                
                # Create working copies of index lists for this module iteration
                current_unselects = list(unselects)
                current_offsprings = list(offsprings)
                
                # 3. Elitism step: Copy Module from Elites to Replacement Slots
                for i in elitist_index:
                    try: 
                        replacee = current_unselects.pop(0)
                    except: 
                        replacee = current_offsprings.pop(0)
                    
                    new_elitists.append(replacee) 
                    # RACE Logic: "self.clone(master=pop[i], replacee=pop[replacee], agent_index=agent_index)"
                    # This only clones the specific module!
                    clone_module(master=pop[i], replacee=pop[replacee], module_index=module_idx)

                # === Compute Agent-Specific Operator Probabilities using Softmax ===
                # Strategy: Use softmax to convert agent_importance into selection probabilities
                # - High importance agents → High probability for CROSSOVER (exploitation)
                # - Low importance agents → High probability for MUTATION (exploration)
                
                # Compute softmax probabilities for crossover selection
                crossover_probs = None
                mutation_probs = None
                
                if agent_importance is not None and module_idx >= 0:
                    # Softmax for crossover: High importance → High probability
                    exp_importance = np.exp(agent_importance - np.max(agent_importance))  # Numerical stability
                    crossover_probs = exp_importance / np.sum(exp_importance)
                    
                    # Inverse softmax for mutation: Low importance → High probability
                    # Use negative importance for inverse relationship
                    exp_inv_importance = np.exp(-agent_importance + np.max(agent_importance))
                    mutation_probs = exp_inv_importance / np.sum(exp_inv_importance)
                
                # 4. Crossover Loop with Probabilistic Module Selection
                if len(current_unselects) % 2 != 0:
                    current_unselects.append(current_unselects[fastrand.pcg32bounded(len(current_unselects))])
                
                for i, j in zip(current_unselects[0::2], current_unselects[1::2]):
                    # Decide whether to apply crossover based on agent importance
                    # If module_idx >= 0 and we have importance, use softmax probability
                    # Otherwise use default high probability
                    should_crossover = True
                    if crossover_probs is not None and module_idx >= 0:
                        # Sample from Bernoulli distribution with probability = crossover_probs[module_idx]
                        should_crossover = random.random() < crossover_probs[module_idx]
                    
                    if should_crossover:
                        off_i = random.choice(new_elitists)
                        off_j = random.choice(current_offsprings)
                        
                        # Inherit Base (Component Only!) - Reset slot to parent's component
                        clone_module(master=pop[off_i], replacee=pop[i], module_index=module_idx)
                        clone_module(master=pop[off_j], replacee=pop[j], module_index=module_idx)
                        
                        # Stochastic Swap (Module Crossover)
                        if random.random() < 0.5:
                            clone_module(master=pop[i], replacee=pop[j], module_index=module_idx)
                        else:
                            clone_module(master=pop[j], replacee=pop[i], module_index=module_idx)

                # 5. Mutation with Probabilistic Module Selection
                for i in range(self.population_size):
                    if i not in new_elitists: # Spare the new_elitists for THIS module
                        # Decide whether to apply mutation based on agent importance
                        # If module_idx >= 0 and we have importance, scale by mutation probability
                        mutation_rate = self.args.mutation_prob
                        if mutation_probs is not None and module_idx >= 0:
                            # Scale base mutation probability by agent-specific mutation preference
                            mutation_rate = mutation_probs[module_idx]
                        
                        if random.random() < mutation_rate:
                            mutate_module(pop[i], module_index=module_idx, args=self.args)
                
                if module_idx == -1: # Use SR elites as the reference "Elites" for return
                     final_elitists_indices.update(new_elitists)

            # Return structure: new_elitists (indices), worst_index (indices)
            # Since elites are mixed, we returnd indices that were protected in the SR pass
            return list(final_elitists_indices), []
        
        # ---------------------------------------------------------------------
        # CASE 2: PARAMETER-LEVEL EVOLUTION (Original/Fallback Logic)
        # ---------------------------------------------------------------------
        worst_index = index_rank[int(-0.5 * len(index_rank)):] # replace half of the individuals
        
        # Selection step
        offsprings = self.selection_tournament(index_rank, num_offsprings=len(index_rank) - self.num_elitists,
                                               tournament_size=3)

        # Figure out unselected candidates
        unselects = []
        new_elitists = []
        for i in range(self.population_size):
            if i not in offsprings and i not in elitist_index:
                unselects.append(i)
        random.shuffle(unselects)

        # Elitism step, assigning elite candidates to some unselects
        for i in elitist_index:
            try:
                replacee = unselects.pop(0)
            except:
                replacee = offsprings.pop(0)
            new_elitists.append(replacee)
            self.clone(master=pop[i], replace=pop[replacee])

        # Crossover between elite and offsprings for the unselected genes with 100 percent probability
        if len(unselects) % 2 != 0:  # Number of unselects left should be even
            unselects.append(unselects[fastrand.pcg32bounded(len(unselects))])
        for i, j in zip(unselects[0::2], unselects[1::2]):
            off_i = random.choice(new_elitists)
            off_j = random.choice(offsprings)
            self.clone(master=pop[off_i], replace=pop[i])
            self.clone(master=pop[off_j], replace=pop[j])

            # Standard Parameter Crossover (Deprecated for Agent-Level but kept for basic models)
            self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.args.mutation_prob:
                    self.mutate_inplace(pop[i], agent_level=False)

        # Recalculate worst_index to exclude positions now occupied by elites
        worst_index = [idx for idx in worst_index if idx not in new_elitists]
        return new_elitists, worst_index


def unsqueeze(array, axis=1):
    if axis == 0:
        return np.reshape(array, (1, len(array)))
    elif axis == 1:
        return np.reshape(array, (len(array), 1))


# ============================================================================
# Agent-Level Evolution Operators (MACO Agent-Level)
# ============================================================================

def crossover_agent_level(genome1, genome2, agent_index):
    """
    Perform agent-level crossover between two genomes.
    Only modifies the specified agent's weight head (W network).
    
    This implements row-level crossover strategy for MACO:
    - Swap entire rows of weight matrices (preserving neuron integrity)
    - Swap corresponding bias elements
    - Leave shared SR network untouched
    
    Args:
        genome1, genome2: Two Genome instances with MACO agent-level MACs
        agent_index: Index of the agent to crossover (0 ~ n_agents-1)
    
    Raises:
        ValueError: If genomes do not support agent-level operations
    """
    if not genome1.has_agent_level_support():
        raise ValueError("Genome does not support agent-level operations. "
                       "Use 'maco_agent_mac' controller for agent-level evolution.")
    
    # Get weight heads for the specified agent
    agent_W1 = genome1.get_agent_W(agent_index)
    agent_W2 = genome2.get_agent_W(agent_index)
    
    # Collect bias parameters first (for matching with weight matrices)
    b_1, b_2 = None, None
    for param1, param2 in zip(agent_W1.parameters(), agent_W2.parameters()):
        W1 = param1.data
        W2 = param2.data
        if len(W1.shape) == 1:  # Bias vector
            b_1 = W1
            b_2 = W2
    
    # Crossover weight matrices (row-level swapping)
    for param1, param2 in zip(agent_W1.parameters(), agent_W2.parameters()):
        W1 = param1.data
        W2 = param2.data
        
        if len(W1.shape) == 2:  # Weight matrix
            num_variables = W1.shape[0]  # Number of output neurons
            num_cross_overs = fastrand.pcg32bounded(num_variables * 2)
            
            for _ in range(num_cross_overs):
                receiver_choice = random.random()
                ind_cr = fastrand.pcg32bounded(W1.shape[0])  # Random row index
                
                if receiver_choice < 0.5:
                    # Swap row from W2 to W1
                    W1[ind_cr, :] = W2[ind_cr, :]
                    if b_1 is not None and b_2 is not None:
                        b_1[ind_cr] = b_2[ind_cr]
                else:
                    # Swap row from W1 to W2
                    W2[ind_cr, :] = W1[ind_cr, :]
                    if b_2 is not None and b_1 is not None:
                        b_2[ind_cr] = b_1[ind_cr]


def mutate_agent_level(genome, agent_index, args):
    """
    Perform agent-level mutation on a genome.
    Only modifies the specified agent's weight head (W network).
    
    Mutation types:
    - Super mutation: Large gaussian noise (strength × 10)
    - Reset: Reinitialize weights to random values
    - Normal mutation: Small gaussian noise
    
    Args:
        genome: Genome instance with MACO agent-level MAC
        agent_index: Index of the agent to mutate (0 ~ n_agents-1)
        args: Arguments containing mutation parameters:
            - mut_strength: Normal mutation strength (default: 0.1)
            - super_mut_strength: Super mutation strength (default: 10.0)
            - super_mut_prob: Probability of super mutation (default: 0.05)
            - reset_prob: Probability of reset (default: 0.1)
            - mut_frac: Fraction of weights to mutate (default: 0.1)
    
    Raises:
        ValueError: If genome does not support agent-level operations
    """
    if not genome.has_agent_level_support():
        raise ValueError("Genome does not support agent-level operations. "
                       "Use 'maco_agent_mac' controller for agent-level evolution.")
    
    agent_W = genome.get_agent_W(agent_index)
    
    # Get mutation parameters (aligned with parameter-level mutation in mutate_inplace)
    mut_strength = 0.1
    super_mut_strength = 10.0
    # Use prob_reset_and_sup from args (same as parameter-level mutation)
    prob_reset_and_sup = getattr(args, 'prob_reset_and_sup', 0.05)
    super_mut_prob = prob_reset_and_sup
    reset_prob = super_mut_prob + prob_reset_and_sup  # Cumulative probability
    mut_frac = getattr(args, 'frac', 0.1)
    
    state_dict = agent_W.state_dict()
    
    for key in state_dict:
        # Skip layer norm parameters
        if is_lnorm_key(key):
            continue
        
        W = state_dict[key]
        
        if len(W.shape) == 2:  # Weight matrix
            num_variables = W.shape[0]
            
            # Agent-level mutation: mutate ALL rows (aligned with RACE action_prob=1.0)
            for row_idx in range(num_variables):
                # Select fraction of weights to mutate
                index_list = random.sample(range(W.shape[1]), int(W.shape[1] * mut_frac))
                
                random_num = random.random()
                
                if random_num < super_mut_prob:
                    # Super mutation: large gaussian noise
                    for ind in index_list:
                        W[row_idx, ind] += random.gauss(0, super_mut_strength * W[row_idx, ind])
                elif random_num < reset_prob:
                    # Reset: reinitialize to random values
                    for ind in index_list:
                        W[row_idx, ind] = random.gauss(0, 1)
                else:
                    # Normal mutation: small gaussian noise
                    for ind in index_list:
                        W[row_idx, ind] += random.gauss(0, mut_strength * W[row_idx, ind])
                
                # Regularization hard limit (use np.clip as in RACE)
                W[row_idx, :] = torch.from_numpy(np.clip(W[row_idx, :].cpu().numpy(), a_min=-1000000, a_max=1000000)).to(W.device)


def clone_agent_level(master, replacee, agent_index):
    """
    Clone a specific agent from master to replacee.
    Only copies the specified agent's weight head (W network).
    Shared SR network remains unchanged.
    
    Args:
        master: Source Genome instance
        replacee: Target Genome instance
        agent_index: Index of the agent to clone (0 ~ n_agents-1)
    
    Raises:
        ValueError: If genomes do not support agent-level operations
    """
    if not master.has_agent_level_support():
        raise ValueError("Genome does not support agent-level operations. "
                       "Use 'maco_agent_mac' controller for agent-level evolution.")
    
    master_W = master.get_agent_W(agent_index)
    replacee_W = replacee.get_agent_W(agent_index)
    
    # Copy parameters from master to replacee
    for target_param, source_param in zip(replacee_W.parameters(), master_W.parameters()):
        target_param.data.copy_(source_param.data)


# Helper Functions for Generalized Module Evolution
def clone_module(master, replacee, module_index):
    """
    Clone a specific module (SR or Agent Head) from master to replacee.
    module_index = -1 -> Clone SR (Shared Representation)
    module_index >= 0 -> Clone Agent Head [module_index]
    """
    if module_index == -1:
        # Clone SR
        if hasattr(master.mac, 'agent_SR'):
             for target_param, source_param in zip(replacee.mac.agent_SR.parameters(), master.mac.agent_SR.parameters()):
                 target_param.data.copy_(source_param.data)
    else:
        # Clone Agent Head
        clone_agent_level(master, replacee, module_index)

def mutate_module(gene, module_index, args):
    """
    Mutate a specific module (SR or Agent Head).
    module_index = -1 -> Mutate SR
    module_index >= 0 -> Mutate Agent Head
    """
    if module_index == -1:
        # Mutate SR
        if hasattr(gene.mac, 'agent_SR'):
             # Create a dummy wrapper or manually mutate parameters
             # Using existing mutation logic implementation style:
             mutate_params(gene.mac.agent_SR.parameters(), args)
    else:
        # Mutate Agent Head
        mutate_agent_level(gene, module_index, args)

def mutate_params(parameters, args):
    """Applying Gaussian mutation to a list of parameters (reusing logic from mutate_inplace)"""
    # Simply reusing the core logic logic would be best, avoiding code duplication
    # Implementing simplified version here for robustness
    
    params_list = list(parameters)
    num_params = len(params_list)
    ssne_probabilities = np.random.uniform(0, 1, num_params) * 2
    
    # Need access to internal data, iter over params directly
    for i, param in enumerate(params_list):
        if len(param.data.shape) == 2: # Weights only
             W = param.data
             ssne_prob = 1.0
             action_prob = ssne_probabilities[i]
             
             if random.random() < ssne_prob:
                num_variables = W.shape[0]
                for index in range(num_variables):
                    if random.random() <= action_prob:
                         index_list = random.sample(range(W.shape[1]), int(W.shape[1] * args.frac))
                         random_num = random.random()
                         if random_num < args.prob_reset_and_sup: # super mut
                             for ind in index_list:
                                 W[index, ind] += random.gauss(0, 1.0 * W[index, ind]) # super_strength=1.0
                         elif random_num < args.prob_reset_and_sup * 2: # reset
                             for ind in index_list:
                                 W[index, ind] = random.gauss(0, 1)
                         else:
                             for ind in index_list:
                                 W[index, ind] += random.gauss(0, 0.1 * W[index, ind]) # mut_strength=0.1
                         
                         W[index, :] = torch.from_numpy(np.clip(W[index, :].cpu().numpy(), a_min=-1000000, a_max=1000000)).to(W.device)

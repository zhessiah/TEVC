import random
import numpy as np
import fastrand
import torch
from components.episode_buffer import EpisodeBatch
from controllers import REGISTRY as mac_REGISTRY
from itertools import chain

def is_lnorm_key(key):
    return key.startswith('lnorm')

class Genome:
    """
    Genome class containing two MAC objects for Double Q-learning style estimation.
    Forward returns min(q1, q2) to prevent Q-value overestimation.
    """
    def __init__(self, args, buffer, groups):
        self.n_agents = args.n_agents
        self.args = args
        
        # Create two separate MACs with the same configuration
        self.mac1 = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        self.mac2 = mac_REGISTRY[args.mac](buffer.scheme, groups, args)
        
        # Use mac1's action selector as the shared selector
        self.action_selector = self.mac1.action_selector
        self.agent_output_type = args.agent_output_type
        self.save_probs = getattr(self.args, 'save_probs', False)
    
    def train(self):
        """Set both agents to training mode."""
        self.mac1.agent.train()
        self.mac2.agent.train()
    def eval(self):
        """Set both agents to evaluation mode."""
        self.mac1.agent.eval()
        self.mac2.agent.eval()
    
    def init_hidden(self, batch_size):
        """Initialize hidden states for both MACs."""
        self.mac1.init_hidden(batch_size)
        self.mac2.init_hidden(batch_size)
    
    def forward(self, ep_batch, t, test_mode=False):
        """
        Forward pass through both MACs and return the minimum Q-values.
        This prevents Q-value overestimation (Double Q-learning style).
        """
        q1 = self.mac1.forward(ep_batch, t, test_mode=test_mode)
        q2 = self.mac2.forward(ep_batch, t, test_mode=test_mode)
        # Return minimum to prevent overestimation
        return torch.min(q1, q2)
    
    def select_actions(self, ep_batch, t_ep, t_env, bs=slice(None), test_mode=False):
        """Select actions using the minimum Q-values from both MACs."""
        avail_actions = ep_batch["avail_actions"][:, t_ep]
        agent_outputs = self.forward(ep_batch, t_ep, test_mode=test_mode)
        chosen_actions = self.action_selector.select_action(agent_outputs[bs], avail_actions[bs], t_env, test_mode=test_mode)
        return chosen_actions
    
    def parameters(self):
        """Return parameters from both MACs."""
        return chain(self.mac1.parameters(), self.mac2.parameters())

    def load_state(self, other_mac):
        """Load state from another Genome."""
        if hasattr(other_mac, 'mac1') and hasattr(other_mac, 'mac2'):
            # Loading from another Genome
            self.mac1.load_state(other_mac.mac1)
            self.mac2.load_state(other_mac.mac2)
        else:
            # Loading from a single MAC (copy to both)
            self.mac1.load_state(other_mac)
            self.mac2.load_state(other_mac)

    def cuda(self):
        """Move both MACs to CUDA."""
        self.mac1.cuda()
        self.mac2.cuda()
    
    def save_models(self, path):
        """Save both MAC models."""
        torch.save(self.mac1.agent.state_dict(), "{}/agent1.th".format(path))
        torch.save(self.mac2.agent.state_dict(), "{}/agent2.th".format(path))
    
    def load_models(self, path):
        """Load both MAC models."""
        self.mac1.agent.load_state_dict(
            torch.load("{}/agent1.th".format(path), map_location=lambda storage, loc: storage)
        )
        self.mac2.agent.load_state_dict(
            torch.load("{}/agent2.th".format(path), map_location=lambda storage, loc: storage)
        )

    def _build_agents(self, input_shape):
        """Build agents for both MACs."""
        self.mac1._build_agents(input_shape)
        self.mac2._build_agents(input_shape)

    def _build_inputs(self, batch, t):
        """Build inputs for agents (delegates to MAC's method)."""
        return self.mac1._build_inputs(batch, t)

    def _get_input_shape(self, scheme):
        """Get input shape for agents (delegates to MAC's method)."""
        return self.mac1._get_input_shape(scheme)
    def eval(self):
        """Set both agents to evaluation mode."""
        self.mac1.agent.eval()
        self.mac2.agent.eval()
    
    def state_dict(self):
        """
        Return combined state dict of both MACs for evolutionary operations.
        Returns a flat dictionary where keys are prefixed with 'mac1.' or 'mac2.'
        
        Important: The returned tensors share storage with the actual parameters,
        so modifications to the returned tensors will affect the model parameters.
        This is required for the mutate_inplace() operation in NN_Evolver.
        """
        state = {}
        # Add mac1 parameters with prefix - use named_parameters to get actual parameter tensors
        for name, param in self.mac1.agent.named_parameters():
            state[f'mac1.{name}'] = param.data
        
        # Add mac2 parameters with prefix
        for name, param in self.mac2.agent.named_parameters():
            state[f'mac2.{name}'] = param.data
        
        return state
    
    def load_state_dict(self, state_dict):
        """
        Load state dict into both MACs.
        Expects keys prefixed with 'mac1.' or 'mac2.'
        """
        mac1_state = {}
        mac2_state = {}
        
        for key, value in state_dict.items():
            if key.startswith('mac1.'):
                # Remove 'mac1.' prefix
                mac1_state[key[5:]] = value
            elif key.startswith('mac2.'):
                # Remove 'mac2.' prefix
                mac2_state[key[5:]] = value
        
        if mac1_state:
            self.mac1.agent.load_state_dict(mac1_state)
        if mac2_state:
            self.mac2.agent.load_state_dict(mac2_state)

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
        # Evaluate the parents

        b_1 = None
        b_2 = None
        for param1, param2 in zip(gene1.parameters(), gene2.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data
            if len(W1.shape) == 1:
                b_1 = W1
                b_2 = W2

        for param1, param2 in zip(gene1.parameters(), gene2.parameters()):
            # References to the variable tensors
            W1 = param1.data
            W2 = param2.data

            if len(W1.shape) == 2:  # Weights no bias
                num_variables = W1.shape[0]
                # Crossover opertation [Indexed by row]
                num_cross_overs = fastrand.pcg32bounded(num_variables * 2)  # Lower bounded on full swaps
                for i in range(num_cross_overs):
                    receiver_choice = random.random()  # Choose which gene to receive the perturbation
                    if receiver_choice < 0.5:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W1[ind_cr, :] = W2[ind_cr, :]
                        b_1[ind_cr] = b_2[ind_cr]
                    else:
                        ind_cr = fastrand.pcg32bounded(W1.shape[0])  #
                        W2[ind_cr, :] = W1[ind_cr, :]
                        b_2[ind_cr] = b_1[ind_cr]

        # Evaluate the children

    def mutate_inplace(self, gene, agent_level=False):
        trials = 5

        mut_strength = 0.1
        num_mutation_frac = 0.1
        super_mut_strength = 10
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
                if agent_level:
                    ssne_prob = 1.0  # ssne_probabilities[i]
                    action_prob = 1.0
                else:
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
        
        if n_objectives == 5:
            # 5-objective case: Symmetrical weighting with Quality-Diversity
            # Optimality side (f[0], f[1]): split w_optimality equally
            # Robustness side (f[2], f[3], f[4]): split w_robustness equally
            w_optimality = 1.0 - alpha_t
            w_robustness = alpha_t
            
            weights = [
                w_optimality / 2.0,  # f[0]: -TD_error (环境拟合)
                w_optimality / 2.0,  # f[1]: confidence_Q (价值自信)
                w_robustness / 3.0,  # f[2]: -adversarial_loss (攻击敏感度)
                w_robustness / 3.0,  # f[3]: adversarial_confidence (鲁棒自信度)
                w_robustness / 3.0   # f[4]: adversarial_novelty (对抗性新颖度)
            ]
        elif n_objectives == 4:
            # 4-objective case: Symmetrical weighting
            # Optimality side (f[0], f[1]): split w_optimality equally
            # Robustness side (f[2], f[3]): split w_robustness equally
            w_optimality = 1.0 - alpha_t
            w_robustness = alpha_t
            
            weights = [
                w_optimality / 2.0,  # f[0]: -TD_error (环境拟合)
                w_optimality / 2.0,  # f[1]: confidence_Q (价值自信)
                w_robustness / 2.0,  # f[2]: -adversarial_loss (攻击敏感度)
                w_robustness / 2.0   # f[3]: adversarial_confidence (鲁棒自信度)
            ]
        elif n_objectives == 3:
            # 3-objective case (backward compatibility): 
            # f[0] = optimality (full weight)
            # f[1], f[2] = robustness (split weight)
            w_optimality = 1.0 - alpha_t
            w_robustness = alpha_t
            weights = [w_optimality, w_robustness / 2.0, w_robustness / 2.0]
        else:
            # Fallback: equal weights
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

    def epoch(self, pop, fitness_evals, agent_level=False, alpha_t=0.0):
        """
        One epoch of evolutionary algorithm with learning-assisted dynamic weighting.
        
        Args:
            pop: Population of Genome objects
            fitness_evals: List of fitness tuples for each individual
            agent_level: Whether to use agent-level operations
            alpha_t: Learning progress weight for adaptive multi-objective optimization
        """
        # Entire epoch is handled with indices; Index rank nets by fitness evaluation (0 is the best after reversing)

        if self.args.Pareto:
            population = [self.fitness_split(fitness) for fitness in fitness_evals]

            pareto_fronts = self.non_dominated_sorting(population, alpha_t=alpha_t)

            # Calculate crowding distance for each front
            crowding_distances = {}
            for front in pareto_fronts:
                crowding_distances.update(self.calculate_crowding_distance(front, population))

            # Selection step: select individuals from Pareto fronts, using crowding distance for tie-breaking
            index_rank = []
            for front in pareto_fronts:
                # sorted_front = sorted(front, key=lambda i: crowding_distances[i], reverse=True)
                # selected.extend(sorted_front[:len(pop) // len(pareto_fronts)])  # Select individuals from each front
                index_rank.extend(front)
            # Elitism: Keep the best individuals (from the first front)
            elitist_index = index_rank[:self.num_elitists]   
        else:
            index_rank = np.argsort(fitness_evals)[::-1]
            elitist_index = index_rank[:self.num_elitists]  # Elitist indexes safeguard
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

        # COMPUTE RL_SELECTION RATE
        # if self.rl_policy is not None:  # RL Transfer happened
        #     self.selection_stats['total'] += 1.0

        #     if self.rl_policy in elitist_index:
        #         self.selection_stats['elite'] += 1.0
        #     elif self.rl_policy in offsprings:
        #         self.selection_stats['selected'] += 1.0
        #     elif self.rl_policy in unselects:
        #         self.selection_stats['discarded'] += 1.0
        #     self.rl_policy = None

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

            if agent_level:
                if random.random() < 0.5:
                    self.clone(master=pop[i], replace=pop[j])
                else:
                    self.clone(master=pop[j], replace=pop[i])
            else:
                self.crossover_inplace(pop[i], pop[j])

        # Mutate all genes in the population except the new elitists
        for i in range(self.population_size):
            if i not in new_elitists:  # Spare the new elitists
                if random.random() < self.args.mutation_prob:
                    self.mutate_inplace(pop[i], agent_level=agent_level)

        # Recalculate worst_index to exclude positions now occupied by elites
        # worst_index should contain the actual worst performers AFTER elite copying
        worst_index = [idx for idx in worst_index if idx not in new_elitists]
        return new_elitists, worst_index


def unsqueeze(array, axis=1):
    if axis == 0:
        return np.reshape(array, (1, len(array)))
    elif axis == 1:
        return np.reshape(array, (len(array), 1))

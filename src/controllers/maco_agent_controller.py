"""
MACO Agent-Level Controller

This controller extends AttackMAC to support agent-level evolution with SR+W decomposition.

Key differences from AttackMAC (which extends BasicMAC):
1. Agent network decomposed into:
   - agent_SR: Shared State Representation (all agents share)
   - agent_W: List of Weight Heads (one per agent, independent)
2. Enables agent-level crossover and mutation for evolutionary algorithms
3. Fully compatible with MACO training pipeline (including Byzantine attacks)

Architecture:
    AttackMAC functionality + SR+W parameter decomposition
"""

from .attack_controller import AttackMAC
from modules.agents import REGISTRY as agent_REGISTRY
import torch as th


class MACOMAC(AttackMAC):
    """
    MACO Agent-Level Multi-Agent Controller.
    
    Inherits all functionality from BasicMAC, but decomposes agent parameters
    into shared (SR) and independent (W) components for agent-level evolution.
    """
    
    def __init__(self, scheme, groups, args):
        """
        Initialize MACO Agent-Level MAC.
        
        This will create:
        - self.agent_SR: Shared state representation network
        - self.agent_W: List of independent weight head networks (n_agents)
        - All other BasicMAC functionality (action selectors, etc.)
        """
        self.n_agents = args.n_agents
        self.args = args
        
        # Get input shape (same as BasicMAC)
        input_shape = self._get_input_shape(scheme)
        
        # Build decomposed agents (SR + W)
        self._build_agents(input_shape)
        
        # Initialize other components (same as BasicMAC)
        self.agent_output_type = args.agent_output_type
        
        from components.action_selectors import REGISTRY as action_REGISTRY
        self.action_selector = action_REGISTRY[args.action_selector](args)
        self.attack_agent_selector = action_REGISTRY[args.attack_agent_selector](args)
        self.save_probs = getattr(self.args, 'save_probs', False)
        
        self.hidden_states = None
    
    def _build_agents(self, input_shape):
        """
        Build decomposed agent networks (SR + W) and create wrapper for Genome compatibility.
        
        Creates:
        - agent_SR: Shared state representation (fc1 + GRU)
        - agent_W: List of independent weight heads (fc2 for each agent)
        - agent: Wrapper object with train()/eval()/state_dict()/load_state_dict() methods
        """
        # Build shared state representation
        agent_SR = agent_REGISTRY["rnn_SR"](input_shape, self.args)
        
        # Build independent weight heads for each agent
        agent_W = []
        for i in range(self.n_agents):
            agent_W.append(agent_REGISTRY["rnn_W"](input_shape, self.args))
        
        # Create wrapper object that provides standard interface for Genome
        class AgentWrapper:
            def __init__(self, SR, W):
                self.SR = SR
                self.W = W
            
            def train(self):
                self.SR.train()
                for w in self.W:
                    w.train()
            
            def eval(self):
                self.SR.eval()
                for w in self.W:
                    w.eval()
            
            def parameters(self):
                """Return all trainable parameters (SR + all W networks)."""
                for param in self.SR.parameters():
                    yield param
                for w in self.W:
                    for param in w.parameters():
                        yield param
            
            def named_parameters(self):
                """Return all named parameters with SR. and W.i. prefixes."""
                for name, param in self.SR.named_parameters():
                    yield f"SR.{name}", param
                for i, w in enumerate(self.W):
                    for name, param in w.named_parameters():
                        yield f"W.{i}.{name}", param
            
            def cuda(self):
                """Move all networks to CUDA."""
                self.SR.cuda()
                for w in self.W:
                    w.cuda()
                return self
            
            def cpu(self):
                """Move all networks to CPU."""
                self.SR.cpu()
                for w in self.W:
                    w.cpu()
                return self
            
            def to(self, device):
                """Move all networks to specified device."""
                self.SR.to(device)
                for w in self.W:
                    w.to(device)
                return self
            
            def state_dict(self):
                state = {}
                for name, param in self.SR.state_dict().items():
                    state[f"SR.{name}"] = param
                for i, w in enumerate(self.W):
                    for name, param in w.state_dict().items():
                        state[f"W.{i}.{name}"] = param
                return state
            
            def load_state_dict(self, state_dict):
                sr_state = {}
                w_states = [{} for _ in range(len(self.W))]
                
                for key, value in state_dict.items():
                    if key.startswith("SR."):
                        sr_state[key.replace("SR.", "")] = value
                    elif key.startswith("W."):
                        parts = key.split(".", 2)
                        agent_idx = int(parts[1])
                        param_name = parts[2]
                        w_states[agent_idx][param_name] = value
                
                self.SR.load_state_dict(sr_state)
                for i, w in enumerate(self.W):
                    w.load_state_dict(w_states[i])
        
        self.agent = AgentWrapper(agent_SR, agent_W)
        
        # Keep direct references for convenience in forward()
        self.agent_SR = agent_SR
        self.agent_W = agent_W
    
    def forward(self, ep_batch, t, test_mode=False):
        """
        Forward pass through decomposed agent networks.
        
        Process:
        1. Build forced inputs (using real executed actions from AttackMAC)
           OR build standard inputs if do_actions not available
        2. Pass through shared SR to get state embeddings
        3. Pass each agent's embedding through its own W network
        4. Apply softmax if needed (same as AttackMAC)
        
        Args:
            ep_batch: Episode batch
            t: Current timestep
            test_mode: Whether in test mode
            
        Returns:
            agent_outs: (batch_size, n_agents, n_actions) agent outputs
        """
        avail_actions = ep_batch["avail_actions"][:, t]
        bs = ep_batch.batch_size
        
        # Build inputs: use forced inputs if available, otherwise standard inputs
        try:
            # Try to use forced inputs (for training with real executed actions)
            agent_inputs = self._build_forced_inputs(ep_batch, t)
        except (ValueError, KeyError):
            # Fall back to standard inputs (for testing or initial episodes)
            agent_inputs = self._build_inputs(ep_batch, t)
        
        # Reshape inputs for batch processing
        # (batch_size, n_agents, input_shape) -> (batch_size * n_agents, input_shape)
        agent_inputs = agent_inputs.reshape(bs * self.n_agents, -1)
        
        if test_mode:
            self.agent_SR.eval()
            for agent_w in self.agent_W:
                agent_w.eval()
        
        # Step 1: Shared state representation
        # All agents pass through the same SR network
        hidden_states_in = self.hidden_states.reshape(bs * self.n_agents, -1)
        shared_embeddings = self.agent_SR(agent_inputs, hidden_states_in)  # (bs*n_agents, hidden_dim)
        
        # Update hidden states (CRITICAL: reshape back to (bs, n_agents, hidden_dim))
        self.hidden_states = shared_embeddings.view(bs, self.n_agents, -1)
        
        # Step 2: Independent weight heads
        # Each agent's embedding goes through its own W network
        shared_embeddings_reshaped = shared_embeddings.view(bs, self.n_agents, -1)
        agent_outs = []
        for i in range(self.n_agents):
            # Extract this agent's embeddings across all batches
            agent_i_embedding = shared_embeddings_reshaped[:, i, :]  # (bs, hidden_dim)
            
            # Pass through this agent's weight head
            agent_i_out = self.agent_W[i](None, agent_i_embedding)  # (bs, n_actions)
            agent_outs.append(agent_i_out.unsqueeze(1))  # (bs, 1, n_actions)
        
        # Concatenate outputs: (bs, n_agents, n_actions)
        agent_outs = th.cat(agent_outs, dim=1)
        
        # Flatten for masking and softmax
        agent_outs_flat = agent_outs.view(bs * self.n_agents, -1)
        
        # Apply softmax if needed (same as AttackMAC)
        if self.agent_output_type == "pi_logits":
            if getattr(self.args, "mask_before_softmax", True):
                # Make unavailable actions very negative
                reshaped_avail_actions = avail_actions.reshape(bs * self.n_agents, -1)
                agent_outs_flat[reshaped_avail_actions == 0] = -1e10
            
            agent_outs_flat = th.nn.functional.softmax(agent_outs_flat, dim=-1)
            
            if not test_mode:
                # Epsilon floor (from AttackMAC)
                epsilon_action_num = agent_outs_flat.size(-1)
                if getattr(self.args, "mask_before_softmax", True):
                    epsilon_action_num = reshaped_avail_actions.sum(dim=1, keepdim=True).float()
                
                agent_outs_flat = ((1 - self.action_selector.epsilon) * agent_outs_flat
                               + th.ones_like(agent_outs_flat) * self.action_selector.epsilon / epsilon_action_num)
                
                if getattr(self.args, "mask_before_softmax", True):
                    agent_outs_flat[reshaped_avail_actions == 0] = 0.0
        
        return agent_outs_flat.view(bs, self.n_agents, -1)
        
    def init_hidden(self, batch_size):
        """
        Initialize hidden states.
        
        Uses agent_SR's init_hidden (same dimension as BasicMAC).
        """
        self.hidden_states = self.agent_SR.init_hidden()
        if self.hidden_states is not None:
            self.hidden_states = self.hidden_states.unsqueeze(0).expand(batch_size, self.n_agents, -1)
    
    def parameters(self):
        """
        Return all trainable parameters (SR + all Ws).
        
        This is used by the learner for gradient updates.
        """
        for param in self.agent_SR.parameters():
            yield param
        for agent_w in self.agent_W:
            for param in agent_w.parameters():
                yield param
    
    def load_state(self, other_mac):
        """
        Load state from another MACO agent-level controller.
        
        Args:
            other_mac: Another MACOMAC instance
        """
        self.agent_SR.load_state_dict(other_mac.agent_SR.state_dict())
        
        for i in range(self.n_agents):
            self.agent_W[i].load_state_dict(other_mac.agent_W[i].state_dict())
    
    def cuda(self):
        """Move all networks to CUDA."""
        self.agent_SR.cuda()
        for agent_w in self.agent_W:
            agent_w.cuda()
    
    def set_attacker(self, attacker):
        self.attacker = attacker
        #set attacker_action_selection params
        self.attack_agent_selector.set_attacker_args(attacker.p_ref, attacker.lamb)
    
    # Inherited from AttackMAC (no changes needed):
    # - select_actions() - supports Byzantine attacks
    # - _build_forced_inputs() - uses real executed actions
    # - _build_inputs() - standard input building
    # - _get_input_shape() - calculate input dimensions

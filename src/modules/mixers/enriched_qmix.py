import torch as th
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from utils.th_utils import orthogonal_init_
from torch.nn import LayerNorm

class Mixer(nn.Module):
    """
    Enriched QMIX Mixer with Q-aware Hypernetwork.
    
    Key Innovation: Hypernet input = [state, agent_Q_values]
    This allows the mixing weights to adapt based on the current Q-value distribution,
    enabling more expressive value factorization.
    
    Standard QMIX: Hypernet(s) → weights
    Enriched QMIX: Hypernet(s, Q_agents) → weights  ← Q-aware adaptation
    """
    def __init__(self, args, abs=True):
        super(Mixer, self).__init__()

        self.args = args
        self.n_agents = args.n_agents
        self.embed_dim = args.mixing_embed_dim
        self.state_dim = int(np.prod(args.state_shape))
        
        # Enriched input: state + all agents' Q-values
        # For 3D input: Q-values are scalars (1 per agent)
        # For 4D input: Q-values are vectors (n_actions per agent)
        self.enriched_input_dim = self.state_dim + self.n_agents  # state + Q_values

        self.abs = abs # monotonicity constraint
        self.qmix_pos_func = getattr(self.args, "qmix_pos_func", "abs")
        
        # Enriched hyper w1 b1 - now takes [state, Q_values] as input
        self.hyper_w1 = nn.Sequential(
            nn.Linear(self.enriched_input_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.n_agents * self.embed_dim)
        )
        self.hyper_b1 = nn.Sequential(
            nn.Linear(self.enriched_input_dim, self.embed_dim)
        )
        
        # Enriched hyper w2 b2 - now takes [state, Q_values] as input
        self.hyper_w2 = nn.Sequential(
            nn.Linear(self.enriched_input_dim, args.hypernet_embed),
            nn.ReLU(inplace=True),
            nn.Linear(args.hypernet_embed, self.embed_dim)
        )
        self.hyper_b2 = nn.Sequential(
            nn.Linear(self.enriched_input_dim, self.embed_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.embed_dim, 1)
        )

        if getattr(args, "use_orthogonal", False):
            for m in self.modules():
                orthogonal_init_(m)

    def forward(self, qvals, states):
        """
        Forward pass with Q-aware hypernetwork.
        
        Args:
            qvals: Agent Q-values
                - 3D: (batch, t, n_agents) - single action per agent
                - 4D: (batch, t, n_agents, n_actions) - all actions
            states: Global state (batch, t, state_dim)
            
        Returns:
            - 3D input → (batch, t, 1): Global Q-value
            - 4D input → (batch, t, n_actions): Global Q-value for each action
        """
        original_shape = qvals.shape
        
        if len(original_shape) == 4:
            # 4D input: (batch, t, n_agents, n_actions)
            # Process each action separately with Q-aware hypernet
            b, t, n_agents, n_actions = original_shape
            
            mixed_outputs = []
            for a in range(n_actions):
                # Extract Q-values for this action across all agents
                qvals_action = qvals[:, :, :, a]  # (batch, t, n_agents)
                
                # Mix with Q-aware hypernet
                mixed_q = self._mix_qvals_enriched(qvals_action, states)
                mixed_outputs.append(mixed_q)
            
            # Stack along action dimension: (batch, t, n_actions)
            result = th.stack(mixed_outputs, dim=-1).squeeze(-2)
            return result
        
        elif len(original_shape) == 3:
            # 3D input: (batch, t, n_agents) - single action Q-values
            return self._mix_qvals_enriched(qvals, states)
        
        else:
            raise ValueError(f"Expected qvals to be 3D or 4D, got shape {original_shape}")
    
    def _mix_qvals_enriched(self, qvals, states):
        """
        Q-aware mixing: Hypernet input includes both state and Q-values.
        
        Innovation: The mixing weights are conditioned on the current Q-value distribution,
        allowing the network to adapt its factorization based on agent Q-values.
        
        Args:
            qvals: (batch, t, n_agents) - Agent Q-values
            states: (batch, t, state_dim) - Global state
            
        Returns:
            (batch, t, 1) - Mixed global Q-value
        """
        b, t, n_agents = qvals.size()
        
        # Reshape for batch processing
        qvals_flat = qvals.reshape(b * t, self.n_agents)  # (b*t, n_agents)
        states_flat = states.reshape(b * t, self.state_dim)  # (b*t, state_dim)
        
        # Enriched input: concatenate state and Q-values
        # This is the key innovation - hypernet now sees Q-value distribution!
        enriched_input = th.cat([states_flat, qvals_flat], dim=-1)  # (b*t, state_dim + n_agents)
        
        # Generate mixing weights conditioned on [state, Q-values]
        # First layer
        w1 = self.hyper_w1(enriched_input).view(-1, self.n_agents, self.embed_dim)  # (b*t, n_agents, embed)
        b1 = self.hyper_b1(enriched_input).view(-1, 1, self.embed_dim)  # (b*t, 1, embed)
        
        # Second layer
        w2 = self.hyper_w2(enriched_input).view(-1, self.embed_dim, 1)  # (b*t, embed, 1)
        b2 = self.hyper_b2(enriched_input).view(-1, 1, 1)  # (b*t, 1, 1)
        
        # Apply monotonicity constraint
        if self.abs:
            w1 = self.pos_func(w1)
            w2 = self.pos_func(w2)
        
        # Forward pass: Q_tot = W2 * ELU(Q_agents @ W1 + b1) + b2
        qvals_reshaped = qvals_flat.unsqueeze(1)  # (b*t, 1, n_agents)
        hidden = F.elu(th.matmul(qvals_reshaped, w1) + b1)  # (b*t, 1, embed)
        y = th.matmul(hidden, w2) + b2  # (b*t, 1, 1)
        
        # Reshape back to original batch/time dimensions
        return y.view(b, t, -1)  # (batch, t, 1)

    def pos_func(self, x):
        if self.qmix_pos_func == "softplus":
            return th.nn.Softplus(beta=self.args.qmix_pos_func_beta)(x)
        elif self.qmix_pos_func == "quadratic":
            return 0.5 * x ** 2
        else:
            return th.abs(x)
        

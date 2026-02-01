"""
MACO Agent-Level Architecture for Evolutionary Multi-Agent Learning

This module implements a decomposed agent architecture for MACO:
- RNNAgent_SR: Shared State Representation network (fc1 + GRU)
- RNNAgent_W: Independent Weight Head network (fc2)

Key Innovation:
- Enables agent-level crossover and mutation in evolutionary algorithms
- Combines parameter sharing (SR) with agent differentiation (W)
- Supports Byzantine fault tolerance through agent-level diversity
"""

import torch.nn as nn
import torch.nn.functional as F


class RNNAgent_SR(nn.Module):
    """
    State Representation Network - Shared across all agents.
    
    This network extracts state embeddings from observations.
    All agents share the same SR to reduce parameter count and encourage
    consistent feature representations.
    
    Architecture:
        obs -> fc1 -> ReLU -> GRU -> hidden_state
    """
    def __init__(self, input_shape, args):
        super(RNNAgent_SR, self).__init__()
        self.args = args
        
        # First layer: observation to hidden
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        
        # Recurrent layer (optional)
        if getattr(args, 'use_rnn', True):
            self.rnn = nn.GRUCell(args.rnn_hidden_dim, args.rnn_hidden_dim)
        else:
            # Fallback to feedforward if RNN disabled
            self.rnn = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        """Initialize hidden state with zeros."""
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        Forward pass through state representation network.
        
        Args:
            inputs: (batch_size * n_agents, input_shape) observations
            hidden_state: (batch_size * n_agents, rnn_hidden_dim) previous hidden state
            
        Returns:
            h: (batch_size * n_agents, rnn_hidden_dim) new hidden state (state embedding)
        """
        x = F.relu(self.fc1(inputs))
        h_in = hidden_state.reshape(-1, self.args.rnn_hidden_dim)
        
        if getattr(self.args, 'use_rnn', True):
            h = self.rnn(x, h_in)
        else:
            h = F.relu(self.rnn(x))
        
        return h


class RNNAgent_W(nn.Module):
    """
    Weight Head Network - Independent for each agent.
    
    This network maps state embeddings to Q-values.
    Each agent has its own independent W network, allowing for
    agent-level differentiation and evolution.
    
    Architecture:
        hidden_state -> fc2 -> Q-values
    """
    def __init__(self, input_shape, args):
        super(RNNAgent_W, self).__init__()
        self.args = args
        
        # Output layer: hidden to Q-values
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.n_actions)
        
        # Optional layer normalization
        if getattr(args, "use_layer_norm", False):
            self.layer_norm = nn.LayerNorm(args.rnn_hidden_dim)

    def forward(self, inputs, shared_state_embedding):
        """
        Forward pass through weight head network.
        
        Args:
            inputs: Not used (kept for API compatibility)
            shared_state_embedding: (batch_size * n_agents, rnn_hidden_dim) state from SR
            
        Returns:
            q: (batch_size * n_agents, n_actions) Q-values
        """
        if getattr(self.args, "use_layer_norm", False):
            q = self.fc2(self.layer_norm(shared_state_embedding))
        else:
            q = self.fc2(shared_state_embedding)
        
        return q


class FFAgent_SR(nn.Module):
    """
    Feedforward State Representation Network (no RNN).
    
    For environments that don't require recurrent processing.
    """
    def __init__(self, input_shape, args):
        super(FFAgent_SR, self).__init__()
        self.args = args
        
        self.fc1 = nn.Linear(input_shape, args.rnn_hidden_dim)
        self.fc2 = nn.Linear(args.rnn_hidden_dim, args.rnn_hidden_dim)

    def init_hidden(self):
        """Initialize hidden state (dummy for FF network)."""
        return self.fc1.weight.new(1, self.args.rnn_hidden_dim).zero_()

    def forward(self, inputs, hidden_state):
        """
        Forward pass through feedforward SR.
        
        Args:
            inputs: Observations
            hidden_state: Not used (kept for API compatibility)
            
        Returns:
            h: State embedding
        """
        x = F.relu(self.fc1(inputs))
        h = F.relu(self.fc2(x))
        return h


class FFAgent_W(nn.Module):
    """
    Feedforward Weight Head Network.
    
    Same as RNNAgent_W but for feedforward architectures.
    """
    def __init__(self, input_shape, args):
        super(FFAgent_W, self).__init__()
        self.args = args
        
        self.fc3 = nn.Linear(args.rnn_hidden_dim, args.n_actions)

    def forward(self, inputs, shared_state_embedding):
        """
        Forward pass through FF weight head.
        
        Args:
            inputs: Not used
            shared_state_embedding: State from SR
            
        Returns:
            q: Q-values
        """
        q = self.fc3(shared_state_embedding)
        return q

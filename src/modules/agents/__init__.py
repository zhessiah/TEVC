REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import NRNNAgent
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent
from .noisy_agents import NoisyRNNAgent
from .rnn_agent_maco import RNNAgent_SR, RNNAgent_W, FFAgent_SR, FFAgent_W

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
REGISTRY["noisy_rnn"] = NoisyRNNAgent

# MACO agent-level decomposition (SR: Shared Representation, W: Weight Head)
REGISTRY["rnn_SR"] = RNNAgent_SR
REGISTRY["rnn_W"] = RNNAgent_W
REGISTRY["ff_SR"] = FFAgent_SR
REGISTRY["ff_W"] = FFAgent_W
from torch import nn
import torch
from torch.distributions import Categorical
import numpy as np

from policies.network import get_MLP

# with inspirations from cleanrl
# the original code uses a two-clause BSD license.
class DiscreteSoftActor(nn.Module):
    def __init__(self,
                 state_dim: int, 
                 action_dim: int, 
                 hidden=[64],
                 device="cpu"
        ):
        super().__init__()
        self.device = device
        self.pi = get_MLP(state_dim, action_dim, hidden, use_relu=True, final_layer_softmax=True)

    def forward(self, x):
        return self.pi(x)

    def get_action(self, x):
        # taken directly from cleanrl/cleanrl/sac_continuous_action.py
        # i don't think the logprob part is intuitive
        probs = self(x)
        dist = Categorical(probs)
        action = dist.sample()
        return action.item(), probs

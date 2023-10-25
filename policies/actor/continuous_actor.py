from torch import nn
import torch
from torch.distributions import Normal, TanhTransform, TransformedDistribution, AffineTransform
import numpy as np

from policies.network import get_MLP

LOG_STD_MAX = 2
LOG_STD_MIN = -5

# with inspirations from cleanrl
# the original code uses a two-clause BSD license.
class ContinuousSoftActor(nn.Module):
    def __init__(self,
                 state_dim: int, 
                 action_dim: int, 
                 act_bias: np.ndarray,
                 act_scale: np.ndarray,
                 hidden_shared=[64],
                 hidden_mean=[64],
                 hidden_std=[64],
                 device="cpu"
        ):
        super().__init__()
        self.device = device
        self.shared_nn = nn.Sequential(
            *get_MLP(state_dim, hidden_shared[-1], hidden_shared[:-1], use_relu=True),
            nn.ReLU(inplace=True)
        ).to(device)
        self.act_mean_nn = get_MLP(hidden_shared[-1], action_dim, hidden_mean[:-1], use_relu=True).to(device)
        self.act_log_std_nn = get_MLP(hidden_shared[-1], action_dim, hidden_std[:-1], use_relu=True).to(device)
        self.act_bias = torch.from_numpy(act_bias).to(device)
        self.act_scale = torch.from_numpy(act_scale).to(device)
    
    def forward(self, x):
        x = self.shared_nn(x)
        act_mean = self.act_mean_nn(x)
        act_logstd = self.act_log_std_nn(x)
        act_logstd = torch.tanh(act_logstd)
        act_logstd = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (act_logstd + 1)  # From SpinUp / Denis Yarat
        return act_mean, act_logstd

    def get_action(self, x):
        # taken directly from cleanrl/cleanrl/sac_continuous_action.py
        # i don't think the logprob part is intuitive
        mean, log_std = self(x)
        std = log_std.exp()
        normal = torch.distributions.Normal(mean, std)
        x_t = normal.rsample()  # for reparameterization trick (mean + std * N(0,1))
        y_t = torch.tanh(x_t)
        action = y_t * self.act_scale + self.act_bias
        log_prob = normal.log_prob(x_t)
        # Enforcing Action Bound
        log_prob -= torch.log(self.act_scale * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        mean = torch.tanh(mean) * self.act_scale + self.act_bias
        return action, log_prob, mean

    
    def get_distribution(self, x):
        # unused for now
        act_mean, act_std = self(x)
        dist = Normal(act_mean, act_std)
        dist_transformed = TransformedDistribution(dist, [
            TanhTransform(),
            AffineTransform(
                loc=self.act_bias,
                scale=self.act_scale
            )
        ])
        return dist_transformed

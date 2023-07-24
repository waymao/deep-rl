# Implementation of DQN
# Inspired by https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95/
# Also Inspired by the code in the tianshou project.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

class DQN(nn.Module):
    """
    Naive DQN
    """
    def __init__(self, 
            nn_module, 
            state_dim, 
            action_dim, 
            lr=1e-3, 
            gamma=0.99, 
            eps=0.05, 
            target_update_freq=10, # target network update frequency
            device="cpu"
        ):
        super().__init__()
        self.model: nn.Module = nn_module.to(device)

        # deep copy the model into the target model which will
        # only be updated once in a while
        self.target_model = deepcopy(nn_module).to(device)
        self.target_model.eval()
        
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        self.gamma = gamma
        self.eps = eps
        self.target_update_freq = target_update_freq
        self.device = device
        self.update_count = 0
        self.state_dim = state_dim
        self.action_dim = action_dim
    
    def forward(self, x):
        if type(x) != torch.Tensor:
            x = torch.Tensor(x)
        if np.random.random() < self.eps:
            return np.random.randint(0, self.action_dim)
        else:
            return self.target_model(x).argmax().item()
    
    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        self.target_model.load_state_dict(self.model.state_dict())
    
    def update(self, batch):
        # state: NxS
        # action: N
        # reward: N
        # next_state: NxS
        # terminated: N, bool
        # truncated: N, bool
        state, action, reward, next_state, terminated, truncated = batch
        state_NS = torch.tensor(state, dtype=torch.float32, device=self.device)
        action_N = torch.tensor(action, dtype=torch.int64, device=self.device)
        reward_N = torch.tensor(reward, dtype=torch.float32, device=self.device)
        next_state_NS = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        terminated_N = torch.tensor(terminated, dtype=torch.float32, device=self.device)

        # q for current state
        q_all_values_NA = self.model(state_NS) # Q values for all actions
        q_values_N1 = torch.gather(q_all_values_NA, 1, action_N.view(-1, 1)) # selected Q
        
        # q for next state using target model. TODO: why?
        q_values_next_NA: torch.Tensor = self.target_model(next_state_NS)
        max_q_values_next_N = q_values_next_NA.max(1)[0]

        # target and loss
        q_values_hat_N = reward_N + self.gamma * max_q_values_next_N * (1 - terminated_N)
        q_values_N = torch.squeeze(q_values_N1)
        dqn_loss = torch.mean(F.mse_loss(q_values_hat_N, q_values_N))

        # back propagate
        self.optimizer.zero_grad()
        dqn_loss.backward()
        self.optimizer.step()

        # update target network if necessary
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.sync_weight()


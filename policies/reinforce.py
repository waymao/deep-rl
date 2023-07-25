# Implementation of REINFORCE
# Inspired by https://hrl.boyuai.com/chapter/2/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E7%AE%97%E6%B3%95
# Also Inspired by the code in the tianshou project.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

class REINFORCE(nn.Module):
    def __init__(self, module: nn.Module, gamma: float, lr=1e-3, device="cpu"):
        super().__init__()
        self.nn_module = module
        self.gamma = gamma
        self.device = device
        self.optimizer = torch.optim.Adam(params=module.parameters(), lr=lr)
    
    def forward(self, x):
        x = torch.tensor(x, device=self.device)
        logits = self.nn_module(x)
        action_list = torch.distributions.Categorical(logits)
        action = action_list.sample()
        return action.item()
    
    def update(self, batch):
        # state: NxS
        # action: N
        # reward: N
        # next_state: NxS
        # terminated: N, bool
        # truncated: N, bool
        states, rewards, actions = batch
        phi_t = 0

        self.optimizer.zero_grad()
        for state, reward, action in zip(reversed(states), reversed(rewards), reversed(actions)):
            state_S = torch.tensor(state, device=self.device)
            # action = torch.tensor(action, device=self.device)
            # reward = torch.tensor(reward, device=self.device)
            phi_t = reward + phi_t * self.gamma
            log_pi = torch.log(self.nn_module(state_S)[action])
            loss = -phi_t * log_pi
            loss.backward()
        self.optimizer.step()


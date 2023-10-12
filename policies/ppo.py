# Implementation of PPO
# Inspired by https://hrl.boyuai.com/



import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

class PPO(nn.Module):
    def __init__(self, 
                 actor_module: nn.Module, 
                 critic_module: nn.Module, 
                 gamma: float=0.98, 
                 lr=1e-3, lr_critic=1e-3, 
                 device="cpu"):
        super().__init__()
        self.actor_module = actor_module
        self.critic_module = critic_module
        self.gamma = gamma
        self.device = device
        self.optimizer = torch.optim.Adam(params=actor_module.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(params=critic_module.parameters(), lr=lr_critic)
        self.I = 1
    
    def forward(self, x):
        x = torch.tensor(x, device=self.device)
        logits = self.actor_module(x)
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
        state = torch.tensor(batch['state'], device=self.device)
        next_state = torch.tensor(batch['next_state'], device=self.device)
        reward = batch['reward']
        action = batch['action']
        terminated = batch['terminated']

        self.optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        critic_target_value = self.critic_module(state)[0]
        with torch.no_grad():
            critic_next_value = self.critic_module(next_state)[0] if not terminated else 0
            delta = reward + self.gamma * critic_next_value - \
                    critic_target_value
        critic_loss = -delta * critic_target_value
        actor_loss = -delta * torch.log(self.actor_module(state)[action])
        self.I *= self.gamma

        critic_loss.backward()
        actor_loss.backward()
        
        self.optimizer.step()
        self.critic_optimizer.step()


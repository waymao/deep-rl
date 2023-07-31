# Implementation of REINFORCE
# Inspired by https://hrl.boyuai.com/chapter/2/%E7%AD%96%E7%95%A5%E6%A2%AF%E5%BA%A6%E7%AE%97%E6%B3%95
# Also Inspired by the code in the tianshou project.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

class ActorCriticEligibilityTrace(nn.Module):
    def __init__(self, 
                 actor_module: nn.Module, 
                 critic_module: nn.Module, 
                 gamma: float=0.98, 
                 lr_actor=1e-3, lr_critic=1e-3, 
                 lambda_actor = 0.5,
                 lambda_critic = 0.5,
                 device="cpu"):
        super().__init__()
        self.gamma = gamma
        self.device = device
        self.actor_module = actor_module.to(device)
        self.critic_module = critic_module.to(device)
        self.z_actor_module = deepcopy(actor_module)
        self.z_critic_module = deepcopy(actor_module)
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.lambda_actor = lambda_actor
        self.lambda_critic = lambda_critic
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

        self.z_actor_module.zero_grad()
        self.z_critic_module.zero_grad()
        critic_target_value = self.z_critic_module(state)[0]
        with torch.no_grad():
            critic_next_value = self.z_critic_module(next_state)[0] if not terminated else 0
            delta = reward + self.gamma * critic_next_value - \
                    critic_target_value
        
        # update z^w and z^\theta
        critic_loss = critic_target_value
        actor_loss = torch.log(self.z_actor_module(state)[action])
        critic_loss.backward()
        actor_loss.backward()
        with torch.no_grad():
            for p in self.z_actor_module.parameters():
                new_value = self.gamma * self.lambda_actor * p + p.grad
                p.copy_(new_value)
            for p in self.z_critic_module.parameters():
                new_value = self.gamma * self.lambda_critic * p + p.grad
                p.copy_(new_value)
        
        # update w and \theta
        with torch.no_grad():
            for p, p_z in zip(self.actor_module.parameters(), self.z_actor_module.parameters()):
                p = p + self.lr_actor * delta * p_z
            for p, p_z in zip(self.critic_module.parameters(), self.z_critic_module.parameters()):
                p = p + self.lr_critic * delta * p_z
        self.I *= self.gamma


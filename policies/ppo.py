# Implementation of PPO
# Inspired by https://hrl.boyuai.com/
# Implements algorithm shown at
#     https://spinningup.openai.com/en/latest/algorithms/ppo.html


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

def compute_adv(gamma: float, lmbda: float, delta_N: torch.Tensor) -> torch.Tensor:
    # Computes the generalized advantage estimator according to 
    # http://arxiv.org/abs/1506.02438
    device = delta_N.device
    delta_N = delta_N.detach().numpy()
    N = delta_N.shape[0]
    curr_adv = 0
    adv_list = np.zeros_like(delta_N)
    for i, delta in enumerate(reversed(delta_N)):
        curr_adv = gamma * lmbda * curr_adv + delta
        adv_list[N - i - 1] = curr_adv
    return torch.from_numpy(adv_list).to(device)

def compute_rew_to_go(gamma: float, rew_N: torch.Tensor) -> torch.Tensor:
    device = rew_N.device
    rew_N = rew_N.detach().numpy()
    N = rew_N.shape[0]
    rew_tg_N = np.zeros_like(rew_N)
    curr_discount_rew = 0
    for i, rew in enumerate(reversed(rew_N)):
        curr_discount_rew = curr_discount_rew * gamma + rew
        rew_tg_N[N - i - 1] = curr_discount_rew
    return torch.from_numpy(rew_tg_N).to(device)


class PPO(nn.Module):
    def __init__(self, 
                 actor_module: nn.Module, 
                 critic_module: nn.Module, 
                 gamma: float=0.98, 
                 lmbda: float=0.95,
                 lr=1e-3, lr_critic=1e-3, 
                 eps=0.2, update_per_train=10,
                 kl_earlystop=0.2,
                 device="cpu"):
        super().__init__()
        self.actor_module = actor_module
        self.critic_module = critic_module
        self.gamma = gamma
        self.lmbda = lmbda
        self.device = device
        self.optimizer = torch.optim.Adam(params=actor_module.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(params=critic_module.parameters(), lr=lr_critic)
        self.eps = eps
        self.kl_earlystop = kl_earlystop
        self.update_per_train = update_per_train
    
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
        state_NS = torch.tensor(batch['state'], device=self.device)
        next_state_NS = torch.tensor(batch['next_state'], device=self.device)
        reward_N = torch.tensor(batch['reward'], device=self.device)
        action_N = torch.tensor(batch['action'], dtype=torch.int64, device=self.device).view(-1, 1)
        terminated_N = torch.tensor(batch['terminated'], dtype=torch.bool, device=self.device)

        with torch.no_grad():
            # compute advantage
            actor_old_N = self.actor_module.forward(state_NS).gather(1, action_N)
            critic_old_value_N = self.critic_module(state_NS)[0]
            critic_old_target_N = reward_N + self.gamma * self.critic_module(next_state_NS)[0] * ~terminated_N
            delta_N = critic_old_target_N - critic_old_value_N
            adv_N = compute_adv(self.gamma, self.lmbda, delta_N)
            # rew_tg_N = compute_rew_to_go(self.gamma, reward_N)

        for _ in range(self.update_per_train):
            ### Update of Actor
            # compute new policy output
            self.optimizer.zero_grad()
            actor_new_N = self.actor_module.forward(state_NS).gather(1, action_N)
            
            # ratio
            ratio_N = actor_new_N / actor_old_N
            kl = torch.sum(torch.log(ratio_N))
            if self.kl_earlystop is not None and kl > self.kl_earlystop:
                # avoid going too far from the last update
                break
            # print("kl:", kl.item())
            clip_adv_N = torch.clamp(ratio_N, 1 - self.eps, 1 + self.eps) * adv_N
            actor_loss = -torch.min(ratio_N * adv_N, clip_adv_N).mean()
            # print(actor_loss.item())
            actor_loss.backward()
            self.optimizer.step()

            ### update of critic
            self.critic_optimizer.zero_grad()
            # critic_loss = torch.mean(F.mse_loss(self.critic_module(state_NS).squeeze(1), rew_tg_N))
            critic_loss = torch.mean(F.mse_loss(self.critic_module(state_NS).squeeze(1), critic_old_target_N))
            critic_loss.backward()
            self.critic_optimizer.step()


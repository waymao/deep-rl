
# Implementation of PPO
# Inspired by https://hrl.boyuai.com/
# Implements algorithm shown at
#     https://spinningup.openai.com/en/latest/algorithms/ppo.html
# got inspirations of the on-policy training procedure from tianshou.


import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from typing import List


SMALL_NUMBER = 1e-8

def split_indices(
    length: int,
    size: int,
    shuffle: bool = True,
    merge_last: bool = False,
) -> List[np.ndarray]:
    # taken from tianshou/data/batch.py
    if size == -1:
        size = length
    assert size >= 1  # size can be greater than length, return whole batch
    indices = np.random.permutation(length) if shuffle else np.arange(length)
    merge_last = merge_last and length % size > 0
    batch_indices = []
    for idx in range(0, length, size):
        if merge_last and idx + size + size >= length:
            batch_indices.append(indices[idx:])
            break
        batch_indices.append(indices[idx : idx + size])
    return batch_indices

def compute_adv(gamma: float, lmbda: float, delta_N: torch.Tensor, terminated_N: torch.Tensor) -> torch.Tensor:
    # Computes the generalized advantage estimator according to 
    # http://arxiv.org/abs/1506.02438
    device = delta_N.device
    delta_N = delta_N.detach().cpu().numpy()
    terminated_N = terminated_N.detach().cpu().numpy()
    N = delta_N.shape[0]
    curr_adv = 0
    adv_list = np.zeros_like(delta_N)
    # discount_N = gamma * lmbda * ~terminated_N
    for i in reversed(range(N)):
        curr_adv = gamma * lmbda * ~terminated_N[i] * curr_adv + delta_N[i]
        adv_list[i] = curr_adv
    return torch.from_numpy(adv_list).to(device)


class PPO(nn.Module):
    """
    PPO Module.
    """
    # NOTE: v == critic. pi == actor.
    def __init__(self, 
                 actor_module: nn.Module, 
                 critic_module: nn.Module, 
                 learning_parameters,
                 logger,
                 device="cpu"):
        super().__init__()

        # extract params TODO allow pass in
        gamma = 0.98
        lmbda = 0.95
        lr_actor = 1e-4
        lr_critic = 1e-3
        eps=0.2
        update_per_train=20
        kl_earlystop=0.8
        batch_size=32
        normalize_advantage = False

        # RL stuff
        self.actor = actor_module
        self.critic = critic_module
        self.gamma = gamma  # discount factor
        self.lmbda = lmbda  # similar to TD-lambda, or TD-n. return discount ratio.
        self.device = device
        self.pi_optim = torch.optim.Adam(params=actor_module.parameters(), lr=lr_actor)
        self.v_optim = torch.optim.Adam(params=critic_module.parameters(), lr=lr_critic)
        self.eps = eps  # clip limit
        self.kl_earlystop = kl_earlystop    # kl threshold for early stopping
        self.update_per_train = update_per_train    # number of passes per train
        self.batch_size = batch_size
        self.normalize_advantage = normalize_advantage
    
    def forward(self, x):
        x = torch.from_numpy(x).to(self.device)
        logits = self.actor(x)
        action_list = torch.distributions.Categorical(logits)
        action = action_list.sample()
        return action.item()
    
    def get_best_action(self, x):
        # alias
        return self.forward(x)
    
    def get_v(self, x):
        return self.critic(x).squeeze(1)
    
    def learn(
            self, 
            s1_NS,
            a_N,
            s2_NS,
            rew_N,
            terminated_N,
        ):
        N, S = s1_NS.shape
        a_N1 = a_N.view(-1, 1)

        with torch.no_grad():
            # compute advantage
            pi_old_N = self.actor.forward(s1_NS).gather(1, a_N1)
            v_old_val_N = self.critic(s1_NS).squeeze(1)

            # use the correct V-value given the LTL.
            v_old_next_val_N = self.critic(s2_NS).squeeze(1)
            # v_old_next_val_N = next_v_vals_CN[2]

            # compute delta using td-1 return.
            v_old_td1ret_N = rew_N + self.gamma * v_old_next_val_N * ~terminated_N
            delta_N = v_old_td1ret_N - v_old_val_N
            
            # compute advantage and td-lambda return.
            adv_N = compute_adv(self.gamma, self.lmbda, delta_N, terminated_N)
            assert adv_N.shape == (N,)
            # replace the old target with lambda return
            v_old_lmdret_N = adv_N + v_old_val_N
            # print((v_old_lmdret_N - v_old_td1ret_N).mean())
            # print("val func mean:", v_old_val_N.mean())
            # print("mean delta:", delta_N.mean().item())
            # print("mean actor last layer weight:", self.actor[-2].weight.mean().item())
            # print("mean actor output:", self.actor(s1_NS).mean(axis=0).detach())
            # print("std actor output:", self.actor(s1_NS).std(axis=0).detach())
            # print("action count:", a_N.unique(return_counts=True))
            # print(v_old_val_N.detach())

        for _ in range(self.update_per_train):
            # randomly shuffle the experience
            batch_indices = split_indices(N, self.batch_size, merge_last=True)

            for i in self.actor.parameters():
                if torch.isnan(i).any():
                    print(i)
                    break
            for idxes in batch_indices:
                # M is the batch size
                s1_batch_MS = s1_NS[idxes]
                a_batch_M1 = a_N1[idxes]
                pi_old_batch_M = pi_old_N[idxes]
                adv_batch_M = adv_N[idxes]

                # advantage normalization
                # taken from tianshou
                if self.normalize_advantage:
                    mean, std = adv_batch_M.mean(), adv_batch_M.std()
                    adv_batch_M = (adv_batch_M - mean) / (std + SMALL_NUMBER)  # per-batch norm

                ### Update of Actor
                # compute new policy output
                self.pi_optim.zero_grad()
                pi_new_batch_M = self.actor.forward(s1_batch_MS).gather(1, a_batch_M1)
                # print("pi:", self.actor.forward(s1_batch_MS).detach())
                
                # ratio
                ratio_batch_M = pi_new_batch_M / pi_old_batch_M
                kl = torch.sum(pi_new_batch_M * torch.log(ratio_batch_M))
                # print("mean ratio:", ratio_batch_M.mean().item(), "; kl:", kl.item())
                if self.kl_earlystop is not None and kl > self.kl_earlystop:
                    # avoid going too far from the last update
                    # print("skipped @", _)
                    continue
                # print("kl:", kl.item())
                clip_adv_M = torch.clamp(ratio_batch_M, 1 - self.eps, 1 + self.eps) * adv_batch_M
                pi_loss = -torch.min(ratio_batch_M * adv_batch_M, clip_adv_M).mean()
                # print(actor_loss.item()) TODO remove
                pi_loss.backward()
                # TODO remove debig
                # with torch.no_grad():
                #     print("mean actor last layer gradient:", self.actor[-2].weight.grad.mean().item())
                self.pi_optim.step()

                ### update of critic
                self.v_optim.zero_grad()
                v_new_val_M = self.critic(s1_batch_MS).squeeze(1)
                # print("new v:", v_new_val_M.mean().item(), "max:", v_new_val_M.max().item())
                # v_loss = 0.5 * torch.sum(F.mse_loss(v_new_val_M, v_old_tgt_N[idxes]))
                v_loss = torch.mean((v_new_val_M - v_old_lmdret_N[idxes]) ** 2)
                v_loss.backward()
                self.v_optim.step()

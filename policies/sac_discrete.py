# Implementation of SAC
# Mostly inspired by Clean RL.
# Also Inspired by the code in the tianshou project, boyuai, and SpinningUp.
# it's different from sac2 in that it applies some methods to make computation
# more stable, plus that when learning pi, we used weighted sum of q instead of
# just gathering selected action.

import torch
from torch import nn
from torch.nn import functional as F
import numpy as np
from copy import deepcopy

class DiscreteSAC(nn.Module):
    """
    Naive DQN
    """
    def __init__(self, 
            q1: nn.Module, 
            q2: nn.Module, 
            pi: nn.Module,
            state_dim, 
            action_dim, 
            lr_q=1e-4, 
            lr_pi=1e-4, 
            lr_alpha=3e-4,
            gamma=0.98, 
            alpha=0.05, # trade off coeff
            policy_update_freq=10, # policy network update frequency
            target_update_freq=10, # target network update frequency
            tau=0.005, # soft update ratio
            start_steps=10, # initial exploration phase, per spinning up
            target_entropy=None,
            auto_alpha=True,
            device="cpu"
        ):
        super().__init__()
        # q1
        self.q1 = q1.to(device)
        self.q1_target = deepcopy(q1).to(device)
        self.q1_target.eval()
        # q2
        self.q2 = q2.to(device)
        self.q2_target = deepcopy(q2).to(device)
        self.q2_target.eval()
        # q optim
        self.q_optim = torch.optim.Adam(list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr_q, eps=1e-4)
        # pi
        self.pi = pi.to(device)
        self.pi_optim = torch.optim.Adam(self.pi.parameters(), lr=lr_pi, eps=1e-4)
        self.tau = tau

        # alpha and autotuning
        self.auto_alpha = auto_alpha
        self.log_alpha = torch.tensor(np.log(alpha), device=device)
        if auto_alpha:
            self.log_alpha.requires_grad = True
            self.target_entropy = target_entropy or -action_dim
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=lr_alpha)
        # hyperparams
        self.gamma = gamma
        self.target_update_freq = target_update_freq
        self.policy_update_freq = policy_update_freq
        self.device = device
        self.update_count = 0
        self.state_dim = state_dim
        self.action_dim = action_dim

        # initial exploration phase
        self.start_steps = start_steps
        self.step = 0
    
    def forward(self, x):
        if self.step <= self.start_steps:
            self.step += 1
            print("random")
            return torch.floor(torch.rand(1) * self.action_dim).int()[0].item()
        else:
            return self.pi.get_action(x)[0][0].item()
    
    def sync_weight(self) -> None:
        """Synchronize the weight for the target network."""
        with torch.no_grad():
            for param, target_param in zip(self.q1.parameters(), self.q1_target.parameters()):
                target_param.copy_(self.tau * param + (1 - self.tau) * target_param)
            for param, target_param in zip(self.q2.parameters(), self.q2_target.parameters()):
                target_param.copy_(self.tau * param + (1 - self.tau) * target_param)
    
    def calc_q_target(self, s2_NS, r_N, ter_N):
        alpha = torch.exp(self.log_alpha)
        _, log_prob_NA, prob_NA = self.pi.get_action(s2_NS)
        q1_next_NA = self.q1_target(s2_NS)
        q2_next_NA = self.q2_target(s2_NS)
        q_min_next_NA = torch.min(q1_next_NA, q2_next_NA) - alpha * log_prob_NA
        q_next_weighed_N = (prob_NA * q_min_next_NA).sum(dim=1)
        y_N = r_N + self.gamma * (1 - ter_N) * q_next_weighed_N
        return y_N

    def update(self, batch):
        self.update_count += 1
        state, action, reward, next_state, terminated, truncated = batch
        s1_NS = torch.tensor(state, dtype=torch.float32, device=self.device)
        a_N = torch.tensor(action, dtype=torch.int64, device=self.device)
        r_N = torch.tensor(reward, dtype=torch.float32, device=self.device)
        s2_NS = torch.tensor(next_state, dtype=torch.float32, device=self.device)
        ter_N = torch.tensor(terminated, dtype=torch.float32, device=self.device)
        alpha = torch.exp(self.log_alpha).detach()

        # q for current state
        q1_val_N = self.q1(s1_NS).gather(1, a_N.view(-1, 1)).squeeze()
        q2_val_N = self.q2(s1_NS).gather(1, a_N.view(-1, 1)).squeeze()

        # q for next state using newly sampled actions.
        with torch.no_grad():
            y_N = self.calc_q_target(s2_NS, r_N, ter_N)

        # q target and loss
        # back propagate q loss
        q_loss1 = torch.mean(F.mse_loss(q1_val_N, y_N))
        q_loss2 = torch.mean(F.mse_loss(q2_val_N, y_N))
        self.q_optim.zero_grad()
        q_loss1.backward()
        q_loss2.backward()
        # print("q", q_loss1.item())
        self.q_optim.step()

        # pi loss
        if self.update_count % self.policy_update_freq == 0:
            for _ in range(self.policy_update_freq):
                _, log_pi_NA, action_probs_NA = self.pi.get_action(s1_NS)
                with torch.no_grad():
                    q1_NA = self.q1(s1_NS)
                    q2_NA = self.q2(s1_NS)
                    min_q_NA = torch.min(q1_NA, q2_NA)
                pi_loss = -torch.mean((min_q_NA - alpha * log_pi_NA) * action_probs_NA)
                self.pi_optim.zero_grad()
                pi_loss.backward()
                self.pi_optim.step()
                
                if self.auto_alpha:
                    alpha_loss = torch.mean(
                        (-log_pi_NA - self.target_entropy)).detach() * torch.exp(self.log_alpha)
                    self.alpha_optim.zero_grad()
                    alpha_loss.backward()
                    self.alpha_optim.step()

        # update target network if necessary
        if self.update_count % self.target_update_freq == 0:
            self.sync_weight()

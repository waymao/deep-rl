from matplotlib import pyplot as plt
import random
import numpy as np
import torch
import gymnasium as gym
from policies.network import get_MLP
from utils.replay_buffer import ReplayBuffer
from torch import nn
from policies.sac import SAC
from policies.actor.continuous_actor import ContinuousSoftActor
env = gym.make('Pendulum-v1', max_episode_steps=200)
env.reset(seed=0)

actor_module = ContinuousSoftActor(state_dim=3, 
                                   action_dim=1, 
                                   act_bias=np.array([0]),
                                   act_scale=np.array([2]))
critic_module = get_MLP(
    num_features=3 + 1,
    num_actions=1,
    hidden_layers=[128]
)
critic2_module = get_MLP(
    num_features=3 + 1,
    num_actions=1,
    hidden_layers=[128]
)
sac_policy = SAC(
    q1=critic_module,
    q2=critic2_module,
    pi=actor_module,
    state_dim=3,
    action_dim=1,
    lr_q=1e-2,
    lr_pi=1e-3,
    lr_alpha=1e-2,
    auto_alpha=True
)
results_sac = []
buffer = ReplayBuffer(capacity=10000)
minimal_size = 100
batch_size = 64

for epi in range(500):
    observation, info = env.reset(seed=0)
    terminated = False
    truncated = False
    epi_len = 0
    total_return = 0

    while not terminated and not truncated:
        action = sac_policy(torch.from_numpy(observation.reshape(1, -1))).detach().squeeze(0).numpy()
        prev_obs = observation
        observation, reward, terminated, truncated, info = env.step(action)
        buffer.add(prev_obs, action, reward, observation, terminated, truncated)
        epi_len += 1
        total_return += reward

        if buffer.size() > minimal_size:
            sampled = buffer.sample(batch_size)
            sac_policy.update(sampled)
    
    print("epi: {}; len: {}; return: {}".format(epi, epi_len, total_return))
    results_sac.append((epi_len, total_return))
env.close()



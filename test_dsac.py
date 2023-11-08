from matplotlib import pyplot as plt
import random
import numpy as np
import torch
import gymnasium as gym
from policies.network import get_MLP
from utils.replay_buffer import ReplayBuffer
from torch import nn
from policies.sac_discrete import DiscreteSAC
from policies.actor.discrete_actor import DiscreteSoftActor
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--env_seed", type=int, default=42)
parser.add_argument("--seed", type=int, default=42)

args = parser.parse_args()
seed = args.seed
env_seed = args.env_seed
print("seed", seed, "; env seed", env_seed)

env = gym.make('CartPole-v1', max_episode_steps=200)
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)
env.reset(seed=env_seed)

actor_module = DiscreteSoftActor(state_dim=4, 
                                   action_dim=2,
                                   hidden=[128])
critic_module = get_MLP(
    num_features=4,
    num_actions=2,
    hidden_layers=[128]
)
critic2_module = get_MLP(
    num_features=4,
    num_actions=2,
    hidden_layers=[128]
)
sac_policy = DiscreteSAC(
    q1=critic_module,
    q2=critic2_module,
    pi=actor_module,
    state_dim=4,
    action_dim=2,
    lr_q=1e-2,
    lr_pi=1e-3,
    # lr_alpha=1e-2,
    # target_entropy=-1,
    # alpha=0.1,
    alpha=0.1,
    start_steps=10,
    auto_alpha=False
)
results_sac = []
buffer = ReplayBuffer(capacity=10000)
minimal_size = 100
batch_size = 64

for epi in range(5000):
    observation, info = env.reset(seed=0)
    terminated = False
    truncated = False
    epi_len = 0
    total_return = 0

    while not terminated and not truncated:
        action = sac_policy(torch.from_numpy(observation.reshape(1, -1)))
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



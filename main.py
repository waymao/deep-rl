import gymnasium as gym
from utils.replay_buffer import ReplayBuffer
from policies.dqn import DQN
from torch import nn

env = gym.make("CartPole-v1", render_mode="human")

buffer = ReplayBuffer(capacity=10000)
module = nn.Sequential(
    nn.Linear(4, 128),
    nn.Linear(128, 2)
)
policy = DQN(
    nn_module=module,
    state_dim=4,
    action_dim=2,
    eps=0.01,
    gamma=0.98,
)

results = []

for step in range(100):
    observation, info = env.reset(seed=42)
    terminated = False
    truncated = False
    epi_len = 0
    total_return = 0

    while not terminated and not truncated:
        action = policy(observation)
        prev_obs = observation
        observation, reward, terminated, truncated, info = env.step(action)
        buffer.add(prev_obs, action, reward, observation, terminated, truncated)
        epi_len += 1
        total_return += reward

        if buffer.size() > 64:
            sampled = buffer.sample(10)
            policy.update(sampled)

        env.render()
    
    print("Reset. len:", epi_len, "; return:", total_return)
    results.append((epi_len, total_return))
env.close()

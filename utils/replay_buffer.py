# Taken From
# https://hrl.boyuai.com/chapter/2/dqn%E7%AE%97%E6%B3%95/
#

import collections
import numpy as np
import random

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = collections.deque(maxlen=capacity)  # 队列,先进先出

    def add(self, state, action, reward, next_state, terminated, truncated):  # 将数据加入buffer
        # state: 1xS
        # action: int
        # reward: int
        # next_state: 1xS
        # terminated: bool
        self.buffer.append((state, action, reward, next_state, terminated, truncated))

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transitions = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, terminated, truncated = zip(*transitions)
        return np.array(state), action, reward, np.array(next_state), terminated, truncated

    def size(self):  # 目前buffer中数据的数量
        return len(self.buffer)

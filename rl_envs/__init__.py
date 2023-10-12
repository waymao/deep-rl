import gymnasium as gym
from .minigrid_1 import SimpleEnv

gym.register("MiniGridEnv-Simple-v0", SimpleEnv)

import random
import gym
import numpy as np


class SnakeWrapper(gym.Wrapper):
    def __init__(self, env):
        super().__init__(env)
        self.env.fps = 60  # Don't know if this does anything

    def flatten(self, state):
        flattened_obs = np.zeros(np.prod(state.shape) * 3)
        for i in range(len(state)):  # Transforms observations into 1-hot-encoded inputs for the network
            for j in range(len(state[i])):
                flattened_obs[(i * 3) + state[i][j]] = 1
        return flattened_obs

    def reset(self):
        return self.flatten(super().reset()), {}

    def step(self, action):
        obs, reward, done, info = self.env.step(action)
        return self.flatten(obs), reward, done, done, info

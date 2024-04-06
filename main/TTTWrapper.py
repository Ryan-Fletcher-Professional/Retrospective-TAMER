import random

import gym
import numpy as np


class TTTWrapper(gym.Wrapper):
    def __init__(self, env, start_mark, human_render=True):
        super().__init__(env)
        self.current_agent_index = 0
        self.current_mark = start_mark
        self.set_start_mark(start_mark)
        self.human_render = human_render

    def reset(self):
        new_set = super().reset()
        self.current_mark = new_set[1]
        return new_set[0], new_set[1]

    def step(self, action):
        self.env.show_turn(self.human_render, self.current_mark)
        obs, reward, done, info = self.env.step(action)
        self.current_mark = obs[1]
        if self.human_render:
            self.env.render()
        flattened_obs = np.zeros(len(obs[0]) * 3)
        for i in range(len(obs[0])):  # Transforms observations into 1-hot-encoded inputs for the network
            flattened_obs[(i * 3) + obs[0][i]] = 1
        return flattened_obs, reward, done, done, info

    def show_result(self, human_render, total_reward):
        self.env.show_result(human_render, self.current_mark, total_reward)

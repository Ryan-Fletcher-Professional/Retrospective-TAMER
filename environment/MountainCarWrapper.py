import math

import gym
import numpy as np


class MountainCarWrapper(gym.Wrapper):
    def __init__(self, env, frame_limit=200, human_render=True, starting_state=None):
        super().__init__(env)
        self.env = self.env.env  # Should bypass default frame limit
        self.current_agent_index = 0  # TODO : Remove???
        self.frame_limit = frame_limit
        self.frames = 0
        self.human_render = human_render
        self.starting_state = starting_state

    def reset(self):
        self.frames = 0
        ret = super().reset()
        if self.starting_state is not None:
            # print("Starting state: ", self.starting_state)
            self.env.env.state = self.starting_state
            # print(self.env.env.state)
        # else:
            # print("State 0: ", self.env.state)
        return self.env.state, ret[1]

    def step(self, action):
        # print(self.env.state)
        if self.starting_state is None:
            self.starting_state = self.env.state.copy()
            # print("SS: ", self.starting_state)
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames += 1
        if (self.frames >= self.frame_limit) and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info

import random
import gym
import numpy as np


class MountainCarWrapper(gym.Wrapper):
    def __init__(self, env, frame_limit=200, human_render=True):
        super().__init__(env)
        self.env = self.env.env  # Should bypass default frame limit
        self.current_agent_index = 0
        self.frame_limit = frame_limit
        self.frames = 0
        self.human_render = human_render
        self.starting_state = None

    def reset(self):
        self.frames = 0
        ret = super().reset()
        if self.starting_state is not None:
            self.env.state = self.starting_state
        return self.env.state, ret[1]

    def step(self, action):
        if self.starting_state is None:
            self.starting_state = self.env.state
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.frames += 1
        if (self.frames >= self.frame_limit) and not terminated:
            truncated = True
        return obs, reward, terminated, truncated, info

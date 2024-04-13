import math
import gym
import numpy as np
import time
from environment.snake_gym_custom.snake_gym_custom.envs.SnakeEnvCustom import SnakeGame


class SnakeWrapper(gym.Wrapper):
    def __init__(self, env, render_mode, frame_limit=200, max_fps=math.inf, starting_state=None):
        super().__init__(env)
        self.max_fps = max_fps
        self.last_step_time = time.time()
        self.frame_limit = frame_limit
        self.frames = 0
        # self.last_frame = np.zeros(super().reset().shape)
        # self.growing = False
        if render_mode is not None:
            self.env.env.render_yn = render_mode
        self.starting_state = starting_state
        if starting_state is not None:
            self.env.state = starting_state

    def flatten(self, state):
        print(state)
        # flattened_obs = np.zeros(np.prod(state.shape) * 3)
        # for i in range(len(state)):  # Transforms observations into 1-hot-encoded inputs for the network
        #     for j in range(len(state[i])):
        #         flattened_obs[(i * 3) + state[i][j]] = 1
        # return flattened_obs
        return np.append(np.reshape(state, -1), np.array(np.unravel_index(np.argmin(state, axis=None), state.shape)))

    def reset(self):
        self.frames = 0
        # self.last_frame = np.zeros(self.last_frame.shape)
        self.growing = False
        ret = self.flatten(super().reset()), {}
        if self.starting_state is not None:
            self.env.state = self.starting_state
        return ret

    def step(self, action):
        while (time.time() - self.last_step_time) < (1 / self.max_fps):
            time.sleep(1 / 1000)
        obs, reward, done, info = self.env.step(action)
        self.env.render()
        self.last_step_time = time.time()
        self.frames += 1
        truncated = False
        if (self.frames >= self.frame_limit) and not done:
            truncated = True
        return self.flatten(obs), reward, done, truncated, info

    def get_invalid_move(self):
        if np.max(self.env.s.last_frame) < 2:
            return None
        for i in range(len(SnakeGame.ACTIONS)):
            action = SnakeGame.ACTIONS[i]
            last_action = self.env.s.last_action
            if ((action[0] != 0) and (last_action[0] != 0) and (action[0] == (last_action[0] * -1))) or\
               ((action[1] != 0) and (last_action[1] != 0) and (action[1] == (last_action[1] * -1))):
                return i

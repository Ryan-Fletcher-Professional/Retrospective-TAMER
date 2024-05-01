import math
import gym
import numpy as np
import time
from environment.GLOBALS import *
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
        self.starting_state = None
        if starting_state is not None:
            self.starting_state = np.reshape(np.array(starting_state), (SNAKE_GRID_DIMS[0], SNAKE_GRID_DIMS[1]))

    def flatten(self, state):
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
        #ret = self.flatten(super().reset()), {}
        super().reset()
        ret = np.zeros(SNAKE_STATE_SIZE), {}
        if self.starting_state is not None:
            self.env.s.last_frame = np.copy(self.starting_state)
            #ret = self.flatten(self.env.s.get_last_frame()), {}
        return ret

    def step(self, action):
        if self.starting_state is None:
            self.starting_state = np.copy(self.env.s.last_frame)
        while (time.time() - self.last_step_time) < (1 / self.max_fps):
            time.sleep(1 / 1000)                #                          {0, 1, 2} - 1
                                                # step with ACTIONS[current + {-1,0,1}] = {turn left, go forward, turn right}
        obs, reward, done, info = self.env.step((self.env.s.last_action_index + (action - 1)) % len(SnakeGame.ACTIONS))
        self.env.render()
        self.last_step_time = time.time()
        self.frames += 1
        truncated = False
        if (self.frames >= self.frame_limit) and not done:
            truncated = True
        # return self.flatten(obs), reward, done, truncated, info
        # https://8thlight.com/insights/qlearning-teaching-ai-to-play-snake : state array under "Game Loop"
        head_pos = np.array(np.unravel_index(np.argmax(self.env.s.last_frame, axis=None), self.env.s.last_frame.shape))
        left_square = obs[(head_pos[0] + SnakeGame.ACTIONS[self.env.s.last_action_index - 1][0]) % obs.shape[0],
                          (head_pos[1] + SnakeGame.ACTIONS[self.env.s.last_action_index - 1][1]) % obs.shape[1]]
        forward_square = obs[(head_pos[0] + SnakeGame.ACTIONS[self.env.s.last_action_index][0]) % obs.shape[0],
                             (head_pos[1] + SnakeGame.ACTIONS[self.env.s.last_action_index][1]) % obs.shape[1]]
        right_square = obs[(head_pos[0] + SnakeGame.ACTIONS[(self.env.s.last_action_index + 1) % len(SnakeGame.ACTIONS)][0]) % obs.shape[0],
                           (head_pos[1] + SnakeGame.ACTIONS[(self.env.s.last_action_index + 1) % len(SnakeGame.ACTIONS)][1]) % obs.shape[1]]
        dirs = np.array([1 if self.env.s.last_action_index == i else 0 for i in range(len(SnakeGame.ACTIONS))])
            # UP, RIGHT, DOWN, LEFT
        food_up = 1 if (self.env.s.get_apple_location()[1] - head_pos[1]) < 0 else 0
        food_right = 1 if (self.env.s.get_apple_location()[0] - head_pos[0]) > 0 else 0
        food_down = 1 if (self.env.s.get_apple_location()[1] - head_pos[1]) > 0 else 0
        food_left = 1 if (self.env.s.get_apple_location()[0] - head_pos[0]) < 0 else 0
        analysis = np.array([left_square > 0, forward_square > 0, right_square > 0,
                             dirs[0], dirs[1],    dirs[2],   dirs[3],
                             food_up, food_right, food_down, food_left])
        return analysis, reward, done, truncated, info

    def get_invalid_move(self):
        # This only applies if not using forward/left/right scheme
        # if np.max(self.env.s.last_frame) < 2:
        #     return None
        # for i in range(len(SnakeGame.ACTIONS)):
        #     action = SnakeGame.ACTIONS[i]
        #     last_action = self.env.s.last_action
        #     if ((action[0] != 0) and (last_action[0] != 0) and (action[0] == (last_action[0] * -1))) or\
        #        ((action[1] != 0) and (last_action[1] != 0) and (action[1] == (last_action[1] * -1))):
        #         return i
        return None

import math
import random
import gym
import numpy as np
import time


class SnakeWrapper(gym.Wrapper):
    def __init__(self, env, frame_limit=200, max_fps=math.inf):
        super().__init__(env)
        self.max_fps = max_fps
        self.last_step_time = time.time()
        self.frame_limit = frame_limit
        self.frames = 0
        self.last_frame = np.zeros(super().reset().shape)
        self.growing = False

    def flatten(self, state):
        flattened_obs = np.zeros(np.prod(state.shape) * 3)
        for i in range(len(state)):  # Transforms observations into 1-hot-encoded inputs for the network
            for j in range(len(state[i])):
                flattened_obs[(i * 3) + state[i][j]] = 1
        return flattened_obs

    def reset(self):
        self.frames = 0
        self.last_frame = np.zeros(self.last_frame.shape)
        self.growing = False
        return self.flatten(super().reset()), {}

    def step(self, action):
        while (time.time() - self.last_step_time) < (1 / self.max_fps):
            time.sleep(1 / 1000)
        obs, reward, done, info = self.env.step(action)
        # new_frame = np.copy(obs)
        # positions = self.env.get_snake_positions()
        # for i in range(len(positions)):
        #     r, c = positions
        #     new_frame[r, c] = len(positions) - i

            # TODO


        # new_frame = np.clip(np.copy(self.last_frame), 0, None)
        # for r in range(new_frame.shape[0]):
        #     for c in range(new_frame.shape[1]):
        #         if (new_frame[r, c] == 0) and (obs[r, c] == 1):
        #             new_frame[r, c] = np.max(self.last_frame) + 1  # set the new head position
        # if (not self.growing) and (np.sum(self.last_frame[self.last_frame > 0]) != 0):  # if last frame snake didn't eat an apple and had a snake
        #     new_frame[new_frame > 0] -= 1  # decrement the snake body values to cut off the previous end of the tail
        # self.growing = False
        # if np.sum(self.last_frame[self.last_frame < 0]) != 0:  # if last frame had an apple
        #     old_apple_r, old_apple_c = np.unravel_index(np.argmin(self.last_frame, axis=None), self.last_frame.shape)
        #     if obs[old_apple_c, old_apple_r] == 1:  # if the old apple was eaten this frame
        #         self.growing = True
        #         print("\n\n\n\n\n\n\n\ngrowing\n\n\n\n\n\n\n\n")
        # new_frame[obs == 2] = -1  # set the current position of the apple
        # self.last_frame = new_frame
        self.last_step_time = time.time()
        self.frames += 1
        truncated = False
        if (self.frames >= self.frame_limit) and not done:
            truncated = True
        return self.flatten(obs), reward, done, truncated, info

    def get_invalid_move(self):
        r, c = np.unravel_index(np.argmax(self.last_frame, axis=None), self.last_frame.shape)
        #  From snake env:
        #   act = [UP, DOWN, LEFT, RIGHT]
        #   UP = (0, -1)
        #   DOWN = (0, 1)
        #   LEFT = (-1, 0)
        #   RIGHT = (1, 0)
        print(self.last_frame)
        up = self.last_frame[r, c - 1]
        if (up > 0) and (up == (self.last_frame[r, c] - 1)):
            return 0
        down = self.last_frame[r, (c + 1) % self.last_frame.shape[1]]
        if (down > 0) and (down == (self.last_frame[r, c] - 1)):
            return 1
        left = self.last_frame[r - 1, c]
        if (left > 0) and (left == (self.last_frame[r, c] - 1)):
            return 2
        right = self.last_frame[(r + 1) % self.last_frame.shape[0], c]
        if (right > 0) and (right == (self.last_frame[r, c] - 1)):
            return 3

import random
import gym
from gym import spaces
import pygame
import numpy as np
from pygame.locals import *


class SnakeEnvCustom(gym.Env):
    metadata = {"render_modes": ["human"], "render_fps": 60}

    def __init__(self):
        self.observation_space = spaces.Box(low=0, high=3, shape=[10, 10])
        self._action_set = [x for x in range(4)]
        self.action_space = spaces.Discrete(4)
        self.window_size = 300
        pygame.init()
        pygame.display.init()
        self.window = pygame.display.set_mode((self.window_size, self.window_size))
        self.clock = pygame.time.Clock()
        self.size = 10
        self.render_yn = "human"
        self.s = SnakeGame(self.size)

    def step(self, action):
        return self.s.step(action)

    def reset(self):
        self.s = SnakeGame(self.size)
        return self.s.get_last_frame()

    def render(self):
        canvas = pygame.Surface((self.window_size, self.window_size))
        canvas.fill((255, 255, 255))
        pix_square_size = (
                self.window_size / self.size
        )  # The size of a single grid square in pixels

        for r in range(self.s.last_frame.shape[0]):
            for c in range(self.s.last_frame.shape[1]):
                color = (255, 255, 255)
                if self.s.last_frame[r, c] == -1:
                    color = (255, 0, 0)
                elif self.s.last_frame[r, c] > 0:
                    color = (0, 0, 0)
                # Now we draw the agent
                pygame.draw.rect(
                    canvas,
                    color,
                    pygame.Rect(
                        pix_square_size * np.asarray((r, c)),
                        (pix_square_size, pix_square_size),
                    ),
                )

        if self.render_yn == "human":
            # The following line copies our drawings from `canvas` to the visible window
            self.window.blit(canvas, canvas.get_rect())
            pygame.display.flip()
            pygame.display.update()

            # We need to ensure that human-rendering occurs at the predefined framerate.
            # The following line will automatically add a delay to keep the framerate stable.
            self.clock.tick(self.metadata["render_fps"])


class SnakeGame:
    UP = np.array((0, -1))
    DOWN = np.array((0, 1))
    LEFT = np.array((-1, 0))
    RIGHT = np.array((1, 0))
    ACTIONS = [UP, RIGHT, DOWN, LEFT]

    def __init__(self, size, human_render=True):
        self.size = size
        grid = np.zeros((10, 10))
        apple_coords = (random.randint(0, size - 1), random.randint(0, size - 1))
        grid[apple_coords[0], apple_coords[1]] = -1
        snake_coords = apple_coords
        while snake_coords == apple_coords:
            snake_coords = (random.randint(0, size - 1), random.randint(0, size - 1))
        grid[snake_coords[0], snake_coords[1]] = 1
        self.last_frame = grid
        self.done = False
        self.human_render = human_render
        self.last_action_index = 0
        self.last_action = SnakeGame.ACTIONS[self.last_action_index]

    def step(self, action):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                self.done = True

        self.last_action = SnakeGame.ACTIONS[action]
        self.last_action_index = action

        head_coords = np.array(np.unravel_index(np.argmax(self.last_frame, axis=None), self.last_frame.shape))

        apple_coords = np.array(np.unravel_index(np.argmin(self.last_frame, axis=None), self.last_frame.shape))
        new_head_coords = (head_coords + self.last_action) % self.size
        new_frame = np.copy(self.last_frame)
        new_head_value = np.max(self.last_frame) + 1

        if new_head_value == (10 * 10) + 1:
            self.done = True
            self.last_frame = new_frame
            return new_frame, new_head_value - 1, True, {"apple": apple_coords}

        if (new_head_coords[0] == apple_coords[0]) and (new_head_coords[1] == apple_coords[1]):
            while new_frame[apple_coords[0], apple_coords[1]] != 0:
                apple_coords = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            new_frame[apple_coords[0], apple_coords[1]] = -1
            new_frame[new_head_coords[0], new_head_coords[1]] = new_head_value
            self.last_frame = new_frame
            return new_frame, new_head_value, False, {"apple": apple_coords}
        elif self.last_frame[new_head_coords[0], new_head_coords[1]] > 1:
            self.last_frame = new_frame
            return new_frame, new_head_value - 1, True, {"apple": apple_coords}
        else:
            new_frame[new_head_coords[0], new_head_coords[1]] = new_head_value
            new_head_value -= 1
            new_frame[new_frame > 0] -= 1
            self.last_frame = new_frame
            return new_frame, new_head_value, False, {"apple": apple_coords}

    def get_last_frame(self):
        return np.copy(self.last_frame)

    def get_apple_location(self):
        return np.array(np.unravel_index(np.argmin(self.last_frame, axis=None), self.last_frame.shape))

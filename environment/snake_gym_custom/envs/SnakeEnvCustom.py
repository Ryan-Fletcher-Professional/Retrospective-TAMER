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
    ACTIONS = [UP, DOWN, LEFT, RIGHT]

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
        self.last_action = SnakeGame.ACTIONS[random.randint(0, 3)]

    def step(self, action):
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                self.done = True

        self.last_action = SnakeGame.ACTIONS[action]

        head_coords = np.array(np.unravel_index(np.argmax(self.last_frame, axis=None), self.last_frame.shape))


        apple_coords = np.array(np.unravel_index(np.argmin(self.last_frame, axis=None), self.last_frame.shape))
        new_head_coords = (head_coords + self.last_action) % self.size
        new_frame = np.copy(self.last_frame)
        new_head_value = np.max(self.last_frame) + 1

        if new_head_value == (10 * 10) + 1:
            self.done = True
            self.last_frame = new_frame
            return new_frame, new_head_value - 1, True, {}

        if (new_head_coords[0] == apple_coords[0]) and (new_head_coords[1] == apple_coords[1]):
            while new_frame[apple_coords[0], apple_coords[1]] != 0:
                apple_coords = (random.randint(0, self.size - 1), random.randint(0, self.size - 1))
            new_frame[apple_coords[0], apple_coords[1]] = -1
            new_frame[new_head_coords[0], new_head_coords[1]] = new_head_value
            self.last_frame = new_frame
            return new_frame, new_head_value, False, {}
        elif self.last_frame[new_head_coords[0], new_head_coords[1]] > 1:
            self.last_frame = new_frame
            return new_frame, new_head_value - 1, True, {}
        else:
            new_frame[new_head_coords[0], new_head_coords[1]] = new_head_value
            new_head_value -= 1
            new_frame[new_frame > 0] -= 1
            self.last_frame = new_frame
            return new_frame, new_head_value, False, {}

    def get_last_frame(self):
        return np.copy(self.last_frame)

    def get_apple_location(self):
        return np.array(np.unravel_index(np.argmin(self.last_frame, axis=None), self.last_frame.shape))




# import random
# import gym
# from gym import spaces
# import pygame
# import numpy as np
# from pygame.locals import *
#
#
# class SnakeEnvCustom(gym.Env):
#     metadata = {"render_modes": ["human"]}
#
#     def __init__(self):
#         self.observation_space = spaces.Box(low=0, high=3, shape=[10, 10])
#         self._action_set = [x for x in range(4)]
#         self.action_space = spaces.Discrete(4)
#         self.s = SnakeGame()
#
#     def step(self, action):
#         state, reward, done, info = self.s.step(action)
#         state = SnakeEnvCustom._process(state)
#         return state, reward, done, info
#
#     def reset(self):
#         self.s = SnakeGame()
#         img = self.s.reset()
#         return SnakeEnvCustom._process(img)
#
#     def render(self, mode='human', close=False):
#         raise NotImplementedError
#
#     @staticmethod
#     def _equals(arr1, arr2):
#         for i in range(len(arr1)):
#             if arr1[i] != arr2[i]:
#                 return False
#         return True
#
#     @staticmethod
#     def _process(img):
#         ret = list(map(list, np.zeros((int(GRID_HEIGHT), int(GRID_WIDTH)))))
#         ret = list(map(lambda x: list(map(int, x)), ret))
#         for i in range(0, SCREEN_HEIGHT, GRIDSIZE):
#             for k in range(0, SCREEN_WIDTH, GRIDSIZE):
#                 if SnakeEnvCustom._equals(img[i][k], [255, 0, 0, 255]):
#                     ret[int(i / 15)][int(k / 15)] = 2
#                 elif SnakeEnvCustom._equals(img[i][k], [0, 0, 0, 255]):
#                     ret[int(i / 15)][int(k / 15)] = 1
#         return np.array(ret)
#
#
# class SnakeGame(object):
#     def __init__(self):
#         self.screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), 0, 32)
#         self.surface = pygame.Surface(self.screen.get_size())
#         self.surface = self.surface.convert()
#         self.surface.fill((255, 255, 255))
#         self.clock = pygame.time.Clock()
#         self.fps = 60
#         self.done = False
#
#         pygame.key.set_repeat(1, 40)
#
#         self.screen.blit(self.surface, (0, 0))
#         pygame.init()
#         self.fpsClock = pygame.time.Clock()
#
#         self.snake = Snake()
#         self.apple = Apple()
#
#     def reset(self):
#         return SnakeGame._get_image(self.surface)
#
#     def step(self, key):
#         length = self.snake.length
#         for event in pygame.event.get():
#             if event.type == QUIT:
#                 pygame.quit()
#                 self.done = True
#
#         act = [UP, DOWN, LEFT, RIGHT]
#         self.snake.point(act[key])
#         self.surface.fill((255, 255, 255))
#         try:
#             self.snake.move()
#         except SnakeException:
#             self.done = True
#         if self.done:
#             state = SnakeGame._get_image(self.surface)
#             return state, length, self.done, {}
#         check_eat(self.snake, self.apple)
#         self.snake.draw(self.surface)
#         self.apple.draw(self.surface)
#         font = pygame.font.Font(None, 36)
#         text = font.render(str(self.snake.length), 1, (10, 10, 10))
#         text_pos = text.get_rect()
#         text_pos.centerx = 20
#         self.surface.blit(text, text_pos)
#         self.screen.blit(self.surface, (0, 0))
#         state = SnakeGame._get_image(self.surface)
#         pygame.display.flip()
#         pygame.display.update()
#         self.fpsClock.tick(self.fps + self.snake.length / 3)
#         return state, self.snake.length, False, {}
#
#     @staticmethod
#     def _get_image(surface):
#         ret = list(map(lambda x: list(x), np.zeros((SCREEN_HEIGHT, SCREEN_WIDTH))))
#         for j in range(SCREEN_HEIGHT):
#             for k in range(SCREEN_WIDTH):
#                 ret[j][k] = surface.get_at((k, j))
#         return np.array(ret)
#
#
# FPS = 60
# SCREEN_WIDTH, SCREEN_HEIGHT = 150, 150
#
# GRIDSIZE = 15
# GRID_WIDTH = SCREEN_WIDTH / GRIDSIZE
# GRID_HEIGHT = SCREEN_HEIGHT / GRIDSIZE
# UP = (0, -1)
# DOWN = (0, 1)
# LEFT = (-1, 0)
# RIGHT = (1, 0)
#
#
# def draw_box(surf, color, pos):
#     r = pygame.Rect((pos[0], pos[1]), (GRIDSIZE, GRIDSIZE))
#     pygame.draw.rect(surf, color, r)
#
#
# class SnakeException(Exception):
#     pass
#
#
# class Snake(object):
#     def __init__(self):
#         self.lose()
#         self.color = (0, 0, 0)
#
#     def get_head_position(self):
#         return self.positions[0]
#
#     def lose(self):
#         self.length = 1
#         self.positions = [((SCREEN_WIDTH / 2), (SCREEN_HEIGHT / 2))]
#         self.direction = random.choice([UP, DOWN, LEFT, RIGHT])
#
#     def point(self, pt):
#         if self.length > 1 and (pt[0] * -1, pt[1] * -1) == self.direction:
#             return
#         else:
#             self.direction = pt
#
#     def move(self):
#         cur = self.positions[0]
#         x, y = self.direction
#         new = (((cur[0] + (x * GRIDSIZE)) % SCREEN_WIDTH), (cur[1] + (y * GRIDSIZE)) % SCREEN_HEIGHT)
#         if len(self.positions) > 2 and new in self.positions[2:]:
#             self.lose()
#             raise SnakeException
#         else:
#             self.positions.insert(0, new)
#             if len(self.positions) > self.length:
#                 self.positions.pop()
#
#     def draw(self, surf):
#         for p in self.positions:
#             draw_box(surf, self.color, p)
#
#
# class Apple(object):
#     def __init__(self):
#         self.position = (0, 0)
#         self.color = (255, 0, 0)
#         self.randomize()
#
#     def randomize(self):
#         self.position = (random.randint(0, GRID_WIDTH - 1) * GRIDSIZE, random.randint(0, GRID_HEIGHT - 1) * GRIDSIZE)
#
#     def draw(self, surf):
#         draw_box(surf, self.color, self.position)
#
#
# def check_eat(snake, apple):
#     if snake.get_head_position() == apple.position:
#         snake.length += 1
#         apple.randomize()

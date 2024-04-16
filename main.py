import random
import os
import sys
import pygame
from environment.mountain_car_practice.teleop import play
from environment.mountain_car_practice.MountainCarWrapperDowngrade import MountainCarWrapperDowngrade
from training import online
from training.offline import offline_wrapper

sys.path.append(os.getcwd() + "/environment")
import time

import gym
import argparse
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
from pynput import keyboard
from environment.gym_tictactoe.gym_tictactoe.env import TicTacToeEnv
from environment.GLOBALS import *
from environment.SnakeWrapper import SnakeWrapper
from environment.TTTWrapper import TTTWrapper
from networks.Network import Net
from scipy.stats import gamma
from datetime import datetime as time
import time as tm
from environment.MountainCarWrapper import MountainCarWrapper
from environment.mountain_car_custom.mountain_car_custom.envs.mountain_car import MountainCarEnvCustom
import tkinter as tk

device = torch.device('cpu')


def display_message_screen(message):
    def on_keypress(event):
        root.destroy()

    def center_window():
        root.update_idletasks()
        # Get the width and height of the window
        width = root.winfo_width()
        height = root.winfo_height()
        # Calculate the center position
        screen_width = root.winfo_screenwidth()
        screen_height = root.winfo_screenheight()
        x = int((screen_width - width) / 2)
        y = int((screen_height - height) / 2)
        root.geometry(f'+{x}+{y}')

    # Create the main window
    root = tk.Tk()
    root.title("Practice")

    # Configure the window to close on any key press
    root.bind("<KeyPress>", on_keypress)

    # Create a label with the provided message
    label = tk.Label(root, text=message, font=('Helvetica', 16), padx=20, pady=20)
    label.pack()

    root.after(0, center_window())

    # Start the GUI event loop
    root.mainloop()


def run_script_practice(num_runs):
    INTRO_MESSAGE =\
        """
        After this screen, you will immediately see a little car in a valley.
        
        Your goal is to move the car to the top of the right side of the valley.
        
        To move the car right, press the right arrow key. To move it left, press the left arrow key.
        
        Think about how you need to move. The car is not strong enough to simply charge all the way up the mountain.
        
        Play the game a few times to get a good feel for it.
        
        Press any key to continue.
        """

    print("RUNNING PRACTICE SCRIPT")

    display_message_screen(INTRO_MESSAGE)

    for _ in range(num_runs):
        mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
        env = MountainCarWrapperDowngrade(gym.make("MountainCar-v0", render_mode='rgb_array'))
        play(env, keys_to_action=mapping)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_plays', default=5, type=int, help="number of play runs to collect")
    parser.add_argument('--mode', default=DEFAULT_MODE, type=str, help="Which game to do. Ignored if using a script.")
    parser.add_argument('--fps', default=10, type=int, help="max fps at which snake game should run")
    parser.add_argument('--frame_limit', default=200, type=int, help="number of frames before episode cuts off")
    parser.add_argument('--feedback_mode', default="online", type=str, help="online/offline : how feedback will be given? Ignored if using a script.")
    parser.add_argument('--script', default=None, type=str, help="practice/live/retrospective/all")
    parser.add_argument('--save_net', default=False, type=bool, help="Whether to save sheckpoint and result networks in live and retrospective scripts.")
    args = parser.parse_args()

    scripts = [args.script]
    if scripts[0] is None:
        net = Net(args.mode)
        for i in range(args.num_plays):
            if args.feedback_mode == "online":
                data = online.collect_live_data(net, env_name=args.mode, frame_limit=args.frame_limit, snake_max_fps=args.fps)
                print("Data for run " + str(i + 1) + ":\n" + str(data))
            elif args.feedback_mode == "offline":
                offline_wrapper(net, env_name=args.mode, frame_limit=args.frame_limit, snake_max_fps=args.fps)
                print("Didn't collect data, but offline run " + str(i + 1) + " complete.")
    elif scripts[0] == 'all':
        scripts = ['practice', 'live', 'retrospective']
        if random.randint(0, 1) == 1:
            scripts[1], scripts[2] = scripts[2], scripts[1]  # To account for IV that is order of experiments
    else:
        for script in scripts:
            if script == 'practice':
                run_script_practice(MOUNTAINCAR_PRACTICE_NUM_RUNS)
            elif script == 'live':
                run_script_live(NUM_PRACTICE_RUNS, NUM_REAL_RUNS, args.save_net)
            elif script == 'retrospective':
                run_script_retrospective(NUM_PRACTICE_RUNS, NUM_REAL_RUNS, args.save_net)

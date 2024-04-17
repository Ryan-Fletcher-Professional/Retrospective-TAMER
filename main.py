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
        if event.keysym == 'Return':
            root.destroy()
        elif event.keysym == 'Escape':
            print("ESCAPING")
            tm.sleep(0.1)
            sys.exit(1)

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
        root.attributes('-topmost', True)
        root.focus_force()
        root.geometry(f'+{x}+{y}')

    # Create the main window
    root = tk.Tk()
    root.title("Practice")

    # Configure the window to close on enter key press
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
        After this screen, you will immediately see the Mountain Car game, which consists of a little car in a valley.
        
        Your goal is to move the car to the top of the right side of the valley.
        
        To move the car right, press the right arrow key. To move it left, press the left arrow key.
        
        Think about how you need to move. The car is not strong enough to simply charge all the way up the mountain.
        
        Play the game a few times to get a good feel for it.
        
        
        Press the Enter key to continue.
        """

    print("RUNNING PRACTICE SCRIPT")

    display_message_screen(INTRO_MESSAGE)

    for _ in range(num_runs):
        mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
        env = MountainCarWrapperDowngrade(gym.make("MountainCar-v0", render_mode='rgb_array'))
        play(env, keys_to_action=mapping)


def run_script_live(num_practice, num_real, save_net, mc_net=None, ttt_net=None, snake_net=None):
    INTRO_MESSAGE_1 =\
        """
        In this experiment, you will be presented with three games:
        Mountain Car, TicTacToe, and Snake.
        We'll explain their rules later.
        
        For each game, your task will be to use modified 'clicker training' to teach an AI how to play the game.
        Don't worry, it's not as hard as it sounds; we made a program that does the heavy lifting for you! All you need to do is
        press the 'c' key when the AI makes a suboptimal move and press the 'v' key when the AI makes an optimal move.
        
        Make sure you give meaningful feedback, but don't think too too hard about it. Focus on actually giving feedback.
        You don't need to give feedback at every time step except in TicTacToe, but games
        that have a rapid pace give you a lot of opportunity to train your AI.
        
        
        Press the Enter key to see the next screen for more instructions.
        """  # TODO : Change 'optimal' to 'optimal or near-optimal'?

    INTRO_MESSAGE_2 = \
        """
        Before each game, you will be presented with a screen that will tell you the rules of
        the game and remind you how to give feedback.
        Once you're ready after that, you can press the Enter key to begin two practice runs.
        After the practice runs, you'll see the waiting screen for your first real training run.
        (We'll remind you which runs are practice and which are real.)
        
        Before each practice or training run, you'll be presented with a waiting screen.
        Once you're ready there, press the Enter key to continue to the run.
        Then you'll see your AI playing the game, and you will give feedback as it does so.
        It will immediately incorporate your feedback and use it to change its behavior.
        
        After all seven runs are completed for all three games, you'll be done!
    
    
        Press the Enter key to go to the first game.
        """

    MC_RULES =\
        """
        You are about to see your AI play the Mountain Car game.
        The goal is for the controller to move the car to the flag at
        the top of the mountain by accelerating the car left or right.
        
        Press the 'c' key when the AI makes a suboptimal move and press the 'v' key when the AI makes an optimal move.
        
        When your AI is accelerating the Mountain Car, it will have a red arrow next to it, pointing
        in the direction of acceleration. This should make it easier to give accurate feedback.
        
        This game moves many times per second. You don't need to give feedback at every time
        step, but the rapid pace means you have a lot of opportunity to train your AI.
        
        
        Press the Enter key to start the first practice run.
        """

    TTT_RULES =\
        """
        You are about to see your AI play TicTacToe.
        The goal is for your AI to write three 'O' characters in a row.
        (The opponent writes one 'X' in before each of your AI's moves.)
        
        Press the 'c' key when the AI makes a suboptimal move and press the 'v' key when the AI makes an optimal move.
        
        The game will wait for you to give feedback after each of your AI's moves.
        
        
        Press the Enter key to start the first practice run.
        """

    SNAKE_RULES =\
        """
        You are about to see your AI play Snake.
        The goal is for the controller to move the head of the snake in one of the four cardinal directions at
        each step, guiding the snake to eat apples without running the snake into its own tail.
        The snake does not die if it hits a wall; it will simply wrap around to the other side of the board.
        
        Press the 'c' key when the AI makes a suboptimal move and press the 'v' key when the AI makes an optimal move.
        
        This game moves several times per second. You don't need to give feedback at every time
        step, but the rapid pace means you have a lot of opportunity to train your AI.
        
        
        Press the Enter key to start the runs.
        """

    WAIT_PRACTICE =\
        """
        Press the Enter key to start the next PRACTICE run.
        """

    WAIT_REAL =\
        """
        Press the Enter key to start the next TRAINING run.
        """

    OUTRO = \
        """
        All done.
        Thanks for participating in our experiment!
        """

    display_message_screen(INTRO_MESSAGE_1)
    display_message_screen(INTRO_MESSAGE_2)

    if mc_net is None:
        mc_net = Net(MOUNTAIN_CAR_MODE)
    display_message_screen(MC_RULES)
    for _ in range(num_practice):
        display_message_screen(WAIT_PRACTICE)
        # live_feedback(copy(mc_net)) TODO : Implement net copy method so that practice runs don't change anything. Run collect live and discard results.
    for i in range(num_real):
        display_message_screen(WAIT_REAL)
        # live_feedback(mc_net)  TODO : Run collect live s.t. it alters net's parameters.
        if save_net:
            pass  # save(mc_net, i + 1)  TODO : Implement save net method

    if ttt_net is None:
        ttt_net = Net(TTT_MODE)
    display_message_screen(TTT_RULES)
    for _ in range(num_practice):
        display_message_screen(WAIT_PRACTICE)
        # live_feedback(copy(ttt_net)) TODO : Implement net copy method so that practice runs don't change anything. Run collect live and discard results.
    for i in range(num_real):
        display_message_screen(WAIT_REAL)
        # live_feedback(ttt_net)  TODO : Run collect live s.t. it alters net's parameters.
        if save_net:
            pass  # save(ttt_net, i + 1)  TODO : Implement save net method

    if snake_net is None:
        snake_net = Net(SNAKE_MODE)
    display_message_screen(SNAKE_RULES)
    for _ in range(num_practice):
        display_message_screen(WAIT_PRACTICE)
        # live_feedback(copy(snake_net)) TODO : Implement net copy method so that practice runs don't change anything. Run collect live and discard results.
    for i in range(num_real):
        display_message_screen(WAIT_REAL)
        # live_feedback(snake_net)  TODO : Run collect live s.t. it alters net's parameters.
        if save_net:
            pass  # save(snake_net, i + 1)  TODO : Implement save net method

    display_message_screen(OUTRO)


def run_script_retrospective(num_practice, num_real, save_net, mc_net=None, ttt_net=None, snake_net=None):
    INTRO_MESSAGE_1 = \
        """
        In this experiment, you will be presented with three games:
        Mountain Car, TicTacToe, and Snake.
        We'll explain their rules later.

        For each game, your task will be to use modified 'clicker training' to teach an AI how to play the game.
        Don't worry, it's not as hard as it sounds; we made a program that does the heavy lifting for you! All you need to do is
        press the 'c' key when the AI makes a suboptimal move and press the 'v' key when the AI makes an optimal move.

        Make sure you give meaningful feedback, but don't think too too hard about it. Focus on actually giving feedback.
        You don't need to give feedback at every time step except in TicTacToe, but games
        that have a rapid pace give you a lot of opportunity to train your AI.


        Press the Enter key to see the next screen for more instructions.
        """  # TODO : Change 'optimal' to 'optimal or near-optimal'?

    INTRO_MESSAGE_2 = \
        """
        Before each game, you will be presented with a screen that will tell you the rules of
        the game and remind you how to give feedback.
        Once you're ready after that, you can press the Enter key to begin a practice run.
        After the practice run, you'll see the waiting screen for your first real training run.
        (We'll remind you which runs are practice and which are real.)

        Before each practice or training run, you'll be presented with a waiting screen.
        Once you're ready there, press the Enter key to continue to the run.
        Then you'll see your AI playing the game. Don't give any feedback yet.
        Once you've seen the entire game, including how well the AI did, you'll be presented with another waiting screen.
        After you press the Enter key to advance past that screen, you'll see the same run again, and now you should give feedback!
        The AI will not actively incorporate your feedback as you're giving it, but the AI will incorporate your feedback
        before the next run.

        After all seven runs are completed for all three games, you'll be done!


        Press the Enter key to go to the first game.
        """

    MC_RULES = \
        """
        You are about to see your AI play the Mountain Car game.
        The goal is for the controller to move the car to the flag at
        the top of the mountain by accelerating the car left or right.

        When giving feedback:
        Press the 'c' key when the AI makes a suboptimal move and press the 'v' key when the AI makes an optimal move.

        When your AI is accelerating the Mountain Car, it will have a red arrow next to it, pointing
        in the direction of acceleration. This should make it easier to give accurate feedback.

        This game moves many times per second. You don't need to give feedback at every time
        step, but the rapid pace means you have a lot of opportunity to train your AI.


        Press the Enter key to start the first practice run.
        """

    TTT_RULES = \
        """
        You are about to see your AI play TicTacToe.
        The goal is for your AI to write three 'O' characters in a row.
        (The opponent writes one 'X' in before each of your AI's moves.)

        When giving feedback:
        Press the 'c' key when the AI makes a suboptimal move and press the 'v' key when the AI makes an optimal move.

        The game will wait for you to give feedback after each of your AI's moves.


        Press the Enter key to start the first practice run.
        """

    SNAKE_RULES = \
        """
        You are about to see your AI play Snake.
        The goal is for the controller to move the head of the snake in one of the four cardinal directions at
        each step, guiding the snake to eat apples without running the snake into its own tail.
        The snake does not die if it hits a wall; it will simply wrap around to the other side of the board.

        When giving feedback:
        Press the 'c' key when the AI makes a suboptimal move and press the 'v' key when the AI makes an optimal move.

        This game moves several times per second. You don't need to give feedback at every time
        step, but the rapid pace means you have a lot of opportunity to train your AI.


        Press the Enter key to start the first practice run.
        """

    WAIT_PRACTICE = \
        """
        Press the Enter key to start the next PRACTICE run.
        Remember, don't give feedback yet; just pay attention to what your AI does.
        """

    WAIT_FEEDBACK =\
        """
        Press the Enter key to see the same run again and to give feedback as your AI moves.
        """

    WAIT_REAL = \
        """
        Press the Enter key to start the next TRAINING run.
        Remember, don't give feedback yet; just pay attention to what your Ai does.
        """

    OUTRO =\
        """
        All done.
        Thanks for participating in our experiment!
        """

    display_message_screen(INTRO_MESSAGE_1)
    display_message_screen(INTRO_MESSAGE_2)

    if mc_net is None:
        mc_net = Net(MOUNTAIN_CAR_MODE)
    display_message_screen(MC_RULES)
    for _ in range(num_practice):
        display_message_screen(WAIT_PRACTICE)
        # run = show_run(copy(mc_net))  TODO : Implement net copy method so that practice runs don't change anything. Implement action replay.
        display_message_screen(WAIT_FEEDBACK)
        # retrospective_feedback(run, copy(mc_net)) TODO : Run collect retrospective and discard results.
    for i in range(num_real):
        display_message_screen(WAIT_REAL)
        # run = show_run(mc_net)  TODO : Implement action replay.
        display_message_screen(WAIT_FEEDBACK)
        # retrospective_feedback(run, mc_net) TODO : Run collect retrospective s.t. it alters net's parameters.
        if save_net:
            pass  # save(mc_net, i + 1)  TODO : Implement save net method

    if ttt_net is None:
        ttt_net = Net(TTT_MODE)
    display_message_screen(TTT_RULES)
    for _ in range(num_practice):
        display_message_screen(WAIT_PRACTICE)
        # run = show_run(copy(ttt_net))  TODO : Implement net copy method so that practice runs don't change anything. Implement action replay.
        display_message_screen(WAIT_FEEDBACK)
        # retrospective_feedback(run, copy(ttt_net)) TODO : Run collect retrospective and discard results.
    for i in range(num_real):
        display_message_screen(WAIT_REAL)
        # run = show_run(ttt_net)  TODO : Implement action replay.
        display_message_screen(WAIT_FEEDBACK)
        # retrospective_feedback(run, ttt_net) TODO : Run collect retrospective s.t. it alters net's parameters.
        if save_net:
            pass  # save(ttt_net, i + 1)  TODO : Implement save net method

    if snake_net is None:
        snake_net = Net(SNAKE_MODE)
    display_message_screen(SNAKE_RULES)
    for _ in range(num_practice):
        display_message_screen(WAIT_PRACTICE)
        # run = show_run(copy(snake_net))  TODO : Implement net copy method so that practice runs don't change anything. Implement action replay.
        display_message_screen(WAIT_FEEDBACK)
        # retrospective_feedback(run, copy(snake_net)) TODO : Run collect retrospective and discard results.
    for i in range(num_real):
        display_message_screen(WAIT_REAL)
        # run = show_run(snake_net)  TODO : Implement action replay.
        display_message_screen(WAIT_FEEDBACK)
        # retrospective_feedback(run, snake_net) TODO : Run collect retrospective s.t. it alters net's parameters.
        if save_net:
            pass  # save(snake_net, i + 1)  TODO : Implement save net method

    display_message_screen(OUTRO)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_plays', default=5, type=int, help="number of play runs to collect")
    parser.add_argument('--mode', default=DEFAULT_MODE, type=str, help="Which game to do. Ignored if using a script.")
    parser.add_argument('--fps', default=10, type=int, help="max fps at which snake game should run")
    parser.add_argument('--frame_limit', default=200, type=int, help="number of frames before episode cuts off")
    parser.add_argument('--feedback_mode', default="online", type=str, help="online/offline : how feedback will be given? Ignored if using a script.")
    parser.add_argument('--script', default=None, type=str, help="practice/live/retrospective/all")
    parser.add_argument('--save_nets', default=False, type=bool, help="Whether to save checkpoint and result networks in live and retrospective scripts.")
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
                run_script_live(NUM_PRACTICE_RUNS_LIVE, NUM_REAL_RUNS, args.save_net)
            elif script == 'retrospective':
                run_script_retrospective(NUM_PRACTICE_RUNS_RETROSPECTIVE, NUM_REAL_RUNS, args.save_net)

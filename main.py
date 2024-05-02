import math
import random
import os
import sys
import pygame
from environment.mountain_car_practice.teleop import play
from environment.mountain_car_practice.MountainCarWrapperDowngrade import MountainCarWrapperDowngrade
from training import online, offline
from training.offline import offline_wrapper
import json

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
from training.train_network import train_network

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


def run_script_practice(num_runs, frame_limit):
    INTRO_MESSAGE =\
        """
        After this screen, you will immediately see the Mountain Car game, which consists of a little car in a valley.
        
        Your goal is to move the car to the top of the right side of the valley.
        
        To move the car right, press the right arrow key. To move it left, press the left arrow key.
        
        Think about how you need to move. The car is not strong enough to simply charge all the way up the mountain.
        
        Play the game a few times to get a good feel for it.
        
        
        Press the Enter key to continue.
        """

    #print("RUNNING PRACTICE SCRIPT")

    display_message_screen(INTRO_MESSAGE)

    for _ in range(num_runs):
        mapping = {(pygame.K_LEFT,): 0, (pygame.K_RIGHT,): 2}
        env = MountainCarWrapperDowngrade(gym.make("MountainCar-v0", render_mode='rgb_array'))
        play(env, keys_to_action=mapping)


def run_script_live(num_practice, fps,
                    mc_load=None, mc_save=None, mc_frame_limit=None, mc_lr=None, do_mc=None, mc_plays=None,
                    mc_trajs=None,
                    ttt_load=None, ttt_save=None, ttt_frame_limit=None, ttt_lr=None, do_ttt=None,
                    ttt_plays=None, ttt_trajs=None,
                    snake_load=None, snake_save=None, snake_frame_limit=None, snake_lr=None, do_snake=None,
                    snake_plays=None, snake_trajs=None):
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
        
        When your AI is accelerating the Mountain Car, it will have a blue arrow next to it, pointing
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

    if do_mc:
        display_message_screen(MC_RULES)
        for _ in range(num_practice):
            display_message_screen(WAIT_PRACTICE)
            mc_net = Net(MOUNTAIN_CAR_MODE)
            if mc_load is not None:
                try:
                    #print("Loading network from", mc_load)
                    mc_net.load_state_dict(torch.load(mc_load))
                except Exception as e:
                    print("Could not load network. Error: ", e)
            logging_data = online.collect_live_data(mc_net, MOUNTAIN_CAR_MODE, mc_frame_limit, fps, True, lr=mc_lr)
            pygame.quit()
        mc_net = Net(MOUNTAIN_CAR_MODE)
        if mc_load is not None:
            try:
                #print("Loading network from", mc_load)
                mc_net.load_state_dict(torch.load(mc_load))
            except Exception as e:
                print("Could not load network. Error: ", e)
        for i in range(mc_plays):
            display_message_screen(WAIT_REAL)
            logging_data = online.collect_live_data(mc_net, MOUNTAIN_CAR_MODE, mc_frame_limit, fps, True, lr=mc_lr, starting_state=mc_trajs[i % len(mc_trajs)][0], trajectory=mc_trajs[i % len(mc_trajs)][1])
            pygame.quit()
            if mc_save is not None:
                torch.save(mc_net, mc_save + f"/live_mc_{i+1}.pt")
                with open(mc_save + f"/logs/live_mc_{i+1}.json", 'w') as fp:
                    json.dump(logging_data, fp)

    if do_ttt:
        display_message_screen(TTT_RULES)
        for _ in range(num_practice):
            display_message_screen(WAIT_PRACTICE)
            ttt_net = Net(TTT_MODE)
            if ttt_load is not None:
                try:
                    #print("Loading network from", ttt_load)
                    ttt_net.load_state_dict(torch.load(ttt_load))
                except Exception as e:
                    print("Could not load network. Error: ", e)
            logging_data = online.collect_live_data(ttt_net, TTT_MODE, ttt_frame_limit, fps, True, lr=ttt_lr)
        ttt_net = Net(TTT_MODE)
        if ttt_load is not None:
            try:
                #print("Loading network from", ttt_load)
                ttt_net.load_state_dict(torch.load(ttt_load))
            except Exception as e:
                print("Could not load network. Error: ", e)
        for i in range(ttt_plays):
            display_message_screen(WAIT_REAL)
            logging_data = online.collect_live_data(ttt_net, TTT_MODE, ttt_frame_limit, fps, True, lr=ttt_lr, trajectory=ttt_trajs[i % len(ttt_trajs)][1])
            if ttt_save is not None:
                torch.save(ttt_net, ttt_save + f"/live_ttt_{i + 1}.pt")
                with open(ttt_save + f"/logs/live_ttt_{i+1}.json", 'w') as fp:
                    json.dump(logging_data, fp)

    if do_snake:
        display_message_screen(SNAKE_RULES)
        for _ in range(num_practice):
            display_message_screen(WAIT_PRACTICE)
            snake_net = Net(SNAKE_MODE)
            if snake_load is not None:
                try:
                    #print("Loading network from", snake_load)
                    snake_net.load_state_dict(torch.load(snake_load))
                except Exception as e:
                    print("Could not load network. Error: ", e)
            logging_data = online.collect_live_data(snake_net, SNAKE_MODE, snake_frame_limit, fps, True, lr=snake_lr)
            pygame.quit()
        snake_net = Net(SNAKE_MODE)
        if snake_load is not None:
            try:
                #print("Loading network from", snake_load)
                snake_net.load_state_dict(torch.load(snake_load))
            except Exception as e:
                print("Could not load network. Error: ", e)
        for i in range(snake_plays):
            display_message_screen(WAIT_REAL)
            logging_data = online.collect_live_data(snake_net, SNAKE_MODE, snake_frame_limit, fps, True, lr=snake_lr, starting_state=snake_trajs[i % len(snake_trajs)][0], trajectory=snake_trajs[i % len(snake_trajs)][1])
            pygame.quit()
            if snake_save is not None:
                torch.save(snake_net, snake_save + f"/live_snake_{i + 1}.pt")
                with open(snake_save + f"/logs/live_snake_{i+1}.json", 'w') as fp:
                    json.dump(logging_data, fp)

    display_message_screen(OUTRO)


def run_script_retrospective(num_practice, fps,
                             mc_load=None, mc_save=None, mc_frame_limit=None, mc_lr=None, do_mc=None, mc_plays=None,
                             mc_trajs=None,
                             ttt_load=None, ttt_save=None, ttt_frame_limit=None, ttt_lr=None, do_ttt=None,
                             ttt_plays=None, ttt_trajs=None,
                             snake_load=None, snake_save=None, snake_frame_limit=None, snake_lr=None, do_snake=None,
                             snake_plays=None, snake_trajs=None):
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
        Once you've seen the entire game, you'll be presented with another waiting screen.
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

        When your AI is accelerating the Mountain Car, it will have a blue arrow next to it, pointing
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

    display_message_screen(INTRO_MESSAGE_1)
    display_message_screen(INTRO_MESSAGE_2)

    if do_mc:
        display_message_screen(MC_RULES)
        for _ in range(num_practice):
            display_message_screen(WAIT_PRACTICE)
            mc_net = Net(MOUNTAIN_CAR_MODE)
            if mc_load is not None:
                try:
                    #print("Loading network from", mc_load)
                    mc_net.load_state_dict(torch.load(mc_load))
                except Exception as e:
                    print("Could not load network. Error: ", e)
            action_history, env = offline.offline_no_feedback_run(mc_net, MOUNTAIN_CAR_MODE, mc_frame_limit, fps, True)
            display_message_screen(WAIT_FEEDBACK)
            indata, outdata, logging_data = offline.offline_collect_feedback(mc_net, MOUNTAIN_CAR_MODE, action_history, mc_frame_limit, fps,
                                                       True, env=env)
            pygame.quit()
            train_network(mc_net, indata, outdata, lr=mc_lr)
        mc_net = Net(MOUNTAIN_CAR_MODE)
        if mc_load is not None:
            try:
                #print("Loading network from", mc_load)
                mc_net.load_state_dict(torch.load(mc_load))
            except Exception as e:
                print("Could not load network. Error: ", e)
        for i in range(mc_plays):
            display_message_screen(WAIT_REAL)
            action_history, env = offline.offline_no_feedback_run(mc_net, MOUNTAIN_CAR_MODE, mc_frame_limit, fps, True, starting_state=mc_trajs[i % len(mc_trajs)][0], trajectory=mc_trajs[i % len(mc_trajs)][1])
            display_message_screen(WAIT_FEEDBACK)
            indata, outdata, logging_data = offline.offline_collect_feedback(mc_net, MOUNTAIN_CAR_MODE, action_history, mc_frame_limit, fps,
                                                               True, env=env)
            pygame.quit()
            train_network(mc_net, indata, outdata, lr=mc_lr)
            if mc_save is not None:
                torch.save(mc_net.state_dict(), mc_save + f"/retrospective_mc_{i+1}.pt")
                with open(mc_save + f"/logs/retrospective_mc_{i+1}.json", 'w') as fp:
                    json.dump(logging_data, fp)

    if do_ttt:
        display_message_screen(TTT_RULES)
        for _ in range(num_practice):
            display_message_screen(WAIT_PRACTICE)
            ttt_net = Net(TTT_MODE)
            if ttt_load is not None:
                try:
                    #print("Loading network from", ttt_load)
                    ttt_net.load_state_dict(torch.load(ttt_load))
                except Exception as e:
                    print("Could not load network. Error: ", e)
            action_history, env = offline.offline_no_feedback_run(ttt_net, TTT_MODE, ttt_frame_limit, fps, True)
            display_message_screen(WAIT_FEEDBACK)
            indata, outdata, logging_data = offline.offline_collect_feedback(ttt_net, TTT_MODE, action_history, ttt_frame_limit, fps,
                                                               True, env=env)
            train_network(ttt_net, indata, outdata, lr=ttt_lr)
        ttt_net = Net(TTT_MODE)
        if ttt_load is not None:
            try:
                #print("Loading network from", ttt_load)
                ttt_net.load_state_dict(torch.load(ttt_load))
            except Exception as e:
                print("Could not load network. Error: ", e)
        for i in range(ttt_plays):
            display_message_screen(WAIT_REAL)
            action_history, env = offline.offline_no_feedback_run(ttt_net, TTT_MODE, ttt_frame_limit, fps, True, trajectory=ttt_trajs[i % len(ttt_trajs)][1])
            display_message_screen(WAIT_FEEDBACK)
            indata, outdata, logging_data = offline.offline_collect_feedback(ttt_net, TTT_MODE, action_history, ttt_frame_limit, fps,
                                                               True, env=env)
            train_network(ttt_net, indata, outdata, lr=ttt_lr)
            if ttt_save is not None:
                torch.save(ttt_net.state_dict(), ttt_save + f"/retrospective_ttt_{i+1}.pt")
                with open(ttt_save + f"/logs/retrospective_ttt_{i+1}.json", 'w') as fp:
                    json.dump(logging_data, fp)

    if do_snake:
        display_message_screen(SNAKE_RULES)
        for _ in range(num_practice):
            display_message_screen(WAIT_PRACTICE)
            snake_net = Net(SNAKE_MODE)
            if snake_load is not None:
                try:
                    #print("Loading network from", snake_load)
                    snake_net.load_state_dict(torch.load(snake_load))
                except Exception as e:
                    print("Could not load network. Error: ", e)
            action_history, env = offline.offline_no_feedback_run(snake_net, SNAKE_MODE, snake_frame_limit, fps, True)
            display_message_screen(WAIT_FEEDBACK)
            indata, outdata, logging_data = offline.offline_collect_feedback(snake_net, SNAKE_MODE, action_history, snake_frame_limit, fps,
                                                               True, env=env)
            pygame.quit()
            train_network(snake_net, indata, outdata, lr=snake_lr)
        snake_net = Net(SNAKE_MODE)
        if snake_load is not None:
            try:
                #print("Loading network from", snake_load)
                snake_net.load_state_dict(torch.load(snake_load))
            except Exception as e:
                print("Could not load network. Error: ", e)
        for i in range(snake_plays):
            display_message_screen(WAIT_REAL)
            action_history, env = offline.offline_no_feedback_run(snake_net, SNAKE_MODE, snake_frame_limit, fps, True, starting_state=snake_trajs[i % len(snake_trajs)][0], trajectory=snake_trajs[i % len(snake_trajs)][1])
            display_message_screen(WAIT_FEEDBACK)
            indata, outdata, logging_data = offline.offline_collect_feedback(snake_net, SNAKE_MODE, action_history, snake_frame_limit, fps,
                                                               True, env=env)
            pygame.quit()
            train_network(snake_net, indata, outdata, lr=snake_lr)
            if snake_save is not None:
                torch.save(snake_net.state_dict(), snake_save + f"/retrospective_snake_{i+1}.pt")
                with open(snake_save + f"/logs/retrospective_snake_{i+1}.json", 'w') as fp:
                    json.dump(logging_data, fp)


def parse_trajs(directory):
    """
        Each input file should be a text file with a single line in the following format:
        [start_state[0], start_state[1], ...]; [action 0, action 1, ...]

        (snake starting state should be flattened; will be automatically reshaped when used)
    """
    if directory is None:
        return None
    ret = []
    i = 0
    while True:
        try:
            with open(directory + "/" + str(i) + ".txt", 'r') as file:
                i += 1
                parts = file.read().split("; ")
                ret.append(([float(arg) for arg in parts[0][1:-1].split(", ")], [int(arg) for arg in parts[1][1:-1].split(", ")]))
        except FileNotFoundError:
            break
    if len(ret) == 0:
        return None
    return ret  # [ ([start state 0], [action 0-0, action 0-1, ...]),
                #   ([start state 1], [action 1-0, action 1-1, ...]),
                #   ... ]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_plays', default=5, type=int, help="number of play runs to collect. Ignored if script is used.")
    parser.add_argument('--mode', default=DEFAULT_MODE, type=str, help="Which game to do. Ignored if using a script.")
    parser.add_argument('--fps', default=7, type=int, help="max fps at which snake game should run")
    parser.add_argument('--frame_limit', default=200, type=int, help="number of frames before episode cuts off")
    parser.add_argument('--feedback_mode', default="online", type=str, help="online/offline : how feedback will be given? Ignored if using a script.")
    parser.add_argument('--script', default=None, type=str, help="practice/live/retrospective/all")
    parser.add_argument('--load_net', default=None, type=str, help="path to a saved network to load. Ignored if script is given.")
    parser.add_argument('--save_net', default=None, type=str, help="directory in which to save the trained network after the game is complete. Ignored if script is given.")
    parser.add_argument('--load_mc', default=None, type=str, help="path to a saved mountaincar network to load")
    parser.add_argument('--save_mc', default=None, type=str, help="directory in which to save the trained MountainCar network after the game is complete")
    parser.add_argument('--load_ttt', default=None, type=str, help="path to a saved tictactoe network to load")
    parser.add_argument('--save_ttt', default=None, type=str, help="directory in which to save the trained tictactoe network after the game is complete")
    parser.add_argument('--load_snake', default=None, type=str, help="path to a saved snake network to load")
    parser.add_argument('--save_snake', default=None, type=str, help="directory in which to save the trained snake network after the game is complete")
    parser.add_argument('--mc_frame_limit', default=200, type=int, help="number of frames before episode of mountaincar cuts off")
    parser.add_argument('--ttt_frame_limit', default=200, type=int, help="number of frames before episode of tictactoe cuts off")
    parser.add_argument('--snake_frame_limit', default=100, type=int, help="number of frames before episode of snake cuts off")
    parser.add_argument('--mc_lr', default=0.2, type=float, help="learning rate for the network for mountaincar")
    parser.add_argument('--ttt_lr', default=0.2, type=float, help="learning rate for the network for ttt")
    parser.add_argument('--snake_lr', default=0.2, type=float, help="learning rate for the network for snake")
    parser.add_argument('--do_mc', default="True", type=str, help="whether to include the mountaincar game in scripted runs")
    parser.add_argument('--do_ttt', default="True", type=str, help="whether to include the tictactoe game in scripted runs")
    parser.add_argument('--do_snake', default="True", type=str, help="whether to include the snake game in scripted runs")
    parser.add_argument('--mc_plays', default=5, type=int, help="number of plays in the script for mountaincar")
    parser.add_argument('--ttt_plays', default=5, type=int, help="number of plays in the script for tictactoe")
    parser.add_argument('--snake_plays', default=3, type=int, help="number of plays in the script for snake")
    parser.add_argument('--mc_trajectory', default=None, type=str, help="manual trajectory directory in the script for mountaincar")
    parser.add_argument('--ttt_trajectory', default=None, type=str, help="manual trajectory directory in the script for tictactoe")
    parser.add_argument('--snake_trajectory', default=None, type=str, help="manual trajectory directory in the script for snake")

    args = parser.parse_args()

    scripts = [args.script]
    if scripts[0] is None:
        lr = args.mc_lr if args.mode == MOUNTAIN_CAR_MODE else (args.ttt_lr if args.mode == TTT_MODE else (args.snake_lr if args.mode == SNAKE_MODE else None))
        net = Net(args.mode)
        if args.load_net is not None:
            try:
                print("Loading network from", args.load_net)
                net.load_state_dict(torch.load(args.load_net))
            except Exception as e:
                print("Could not load network. Error: ", e)
        for i in range(args.num_plays):
            if args.feedback_mode == "online":
                data = online.collect_live_data(net, env_name=args.mode, frame_limit=args.frame_limit, snake_max_fps=args.fps)
                print("Online run " + str(i + 1) + " complete")
            elif args.feedback_mode == "offline":
                offline_wrapper(net, env_name=args.mode, frame_limit=args.frame_limit, snake_max_fps=args.fps, iter=args.num_plays, lr=lr)
                print("Offline run " + str(i + 1) + " complete.")
        # save the resulting network
        if args.save_net is not None:
            print("Saving trained network at ", args.save_net)
            torch.save(net.state_dict(), args.save_net)
    elif scripts[0] == 'all':
        scripts = ['practice', 'live', 'retrospective']
        if random.randint(0, 1) == 1:
            scripts[1], scripts[2] = scripts[2], scripts[1]  # To account for independent variable that is order of experiments
    if scripts[0] is not None:
        for script in scripts:
            if script == 'practice':
                run_script_practice(MOUNTAINCAR_PRACTICE_NUM_RUNS, args.frame_limit)
            elif script == 'live':
                run_script_live(NUM_PRACTICE_RUNS_LIVE, args.fps,
                                mc_load=args.load_mc, mc_save=args.save_mc, mc_frame_limit=args.mc_frame_limit,
                                    mc_lr=args.mc_lr, do_mc=args.do_mc == "True", mc_plays=args.mc_plays,
                                    mc_trajs=parse_trajs(args.mc_trajectory),
                                ttt_load=args.load_ttt, ttt_save=args.save_ttt, ttt_frame_limit=args.ttt_frame_limit,
                                    ttt_lr=args.ttt_lr, do_ttt=args.do_ttt == "True", ttt_plays=args.ttt_plays,
                                    ttt_trajs=parse_trajs(args.ttt_trajectory),
                                snake_load=args.load_snake, snake_save=args.save_snake,
                                    snake_frame_limit=args.snake_frame_limit, snake_lr=args.snake_lr,
                                    do_snake=args.do_snake == "True", snake_plays=args.snake_plays,
                                    snake_trajs=parse_trajs(args.snake_trajectory))
            elif script == 'retrospective':
                # print(parse_trajs(args.mc_trajectory))
                run_script_retrospective(NUM_PRACTICE_RUNS_RETROSPECTIVE, args.fps,
                                         mc_load=args.load_mc, mc_save=args.save_mc, mc_frame_limit=args.mc_frame_limit,
                                            mc_lr=args.mc_lr, do_mc=args.do_mc == "True", mc_plays=args.mc_plays,
                                            mc_trajs=parse_trajs(args.mc_trajectory),
                                         ttt_load=args.load_ttt, ttt_save=args.save_ttt,
                                            ttt_frame_limit=args.ttt_frame_limit,
                                            ttt_lr=args.ttt_lr, do_ttt=args.do_ttt == "True", ttt_plays=args.ttt_plays,
                                            ttt_trajs=parse_trajs(args.ttt_trajectory),
                                         snake_load=args.load_snake, snake_save=args.save_snake,
                                            snake_frame_limit=args.snake_frame_limit, snake_lr=args.snake_lr,
                                            do_snake=args.do_snake == "True", snake_plays=args.snake_plays,
                                            snake_trajs=parse_trajs(args.snake_trajectory))
        OUTRO = \
            """
            All done.
            Thanks for participating in our experiment!
            """
        display_message_screen(OUTRO)

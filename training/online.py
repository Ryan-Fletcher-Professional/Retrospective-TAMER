import random
import sys
import time
import gym
import argparse
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pynput import keyboard
import environment.snake_gym
from environment.gym_tictactoe.gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark
from environment.GLOBALS import *
from environment.SnakeWrapper import SnakeWrapper
from environment.TTTWrapper import TTTWrapper
from networks.Network import Net
from scipy.stats import gamma
from datetime import datetime as time
import time as tm
from environment.MountainCarWrapper import MountainCarWrapper

from training.format_data import make_state_action, make_training_data, make_training_data_with_gamma
from training.train_network import train_network



def collect_live_data(net, env_name, frame_limit=200, snake_max_fps=20, human_render=True):
    '''
    Run a live simulation and collect keyboard data
    inputs: policy, gym environment name, and whether to render
    outputs: a list of dictionaries {"state": np.array, "feedback": binary}
    '''

    last_action = 0
    last_state = np.array([])
    state_action_history = np.array([])
    feedback_history = np.array([])
    time_data = np.array([])
    full_obs_data = []
    can_go = True

    def on_press(key):
        nonlocal state_action_history, feedback_history, can_go
        if len(full_obs_data) == 0:
            print("Slow down! Haven't even started playing yet.")
            return
        try:
            # c for negative feedback
            if key.char == 'c':
                feedback = 0
            # v for positive feedback
            elif key.char == 'v':
                feedback = 1
            else:
                print("WRONG KEY! Press 'c' or 'v'")
                return
        except Exception as e:
            print("WRONG INPUT! Press 'c' or 'v'")
            print("Exception: ", e)
            return

        state_action = make_state_action(last_state, last_action, env_name)
        state_action_history = np.append(state_action_history, state_action)
        feedback_history = np.append(feedback_history, feedback)
        if env_name == MOUNTAIN_CAR_MODE:
            # how much time passed since first frame
            # and the time the  feedback was recorded
            feedback_delta = (time.now() - start_time).total_seconds()
            input_tensor, output_tensor = \
                make_training_data_with_gamma(full_obs_data, feedback, time_data, feedback_delta)
            # TODO : Train snake w/gamma
        else:
            input_tensor, output_tensor = \
                make_training_data(state_action, feedback)
        # train network
        train_network(net, input_tensor, output_tensor)

        can_go = True

    # start a keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    def get_invalid_actions(inputs):
        invalids = np.zeros(ACTION_SIZES[env_name])
        if env_name == TTT_MODE:
            for j in range(len(invalids)):
                invalids[j] = 1 if inputs[j * 3] == 0 else 0  # First logit for each square is the 'empty square' logit
        if env_name == SNAKE_MODE:
            idx = env.get_invalid_move()
            print(idx)
            if idx is not None:
                invalids[idx] = 1
        if env_name == MOUNTAIN_CAR_MODE:
            # make it so the car has to go left or right.
            invalids[1] = 1

        return invalids

    agents = [lambda net_input: net.predict_max_action(net_input, get_invalid_actions(net_input))]
    current_agent_index = 0
    wait_agents = [0]  # Which agents to wait for when not running continuously

    run_continuously = True

    if env_name == MOUNTAIN_CAR_MODE:
        if human_render:
            env = MountainCarWrapper(gym.make(MOUNTAIN_CAR_MODE, render_mode='human'), frame_limit)
        else:
            env = MountainCarWrapper(gym.make(MOUNTAIN_CAR_MODE), frame_limit)
    elif env_name == TTT_MODE:
        # Wraps the TTT environment to alter arguments for version compatibility
        env = TTTWrapper(TicTacToeEnv(), '0', frame_limit, human_render)
        agents.append(lambda _: random.choice(env.available_actions()))
        run_continuously = False
    elif env_name == SNAKE_MODE:
        env = SnakeWrapper(gym.make(env_name), frame_limit, snake_max_fps)  # Snake game does not use env.render() so we can't make it not render
    elif env_name == "snake-v0":
        print("!!!!!!!!\tWARNING: Do you mean to play snake-tiled-v0?\t!!!!!!!!")
    else:
        print("!!!!!!!!\tError: No valid environment name\t!!!!!!!!")

    total_reward = 0
    last_state, _ = env.reset()
    # play the game until terminated
    done = False
    start_time = time.now()
    while not done:
        # take the action that the network assigns the highest logit value to
        # Note that first we convert from numpy to tensor and then we get the value of the
        # argmax using .item() and feed that into the environment
        current_agent_index = (current_agent_index + 1) % len(agents)  # len(agents) = 1 when in single-agent game
        last_action = agents[current_agent_index](last_state)
        # print(action)
        last_state, current_reward, terminated, truncated, info = env.step(last_action)
        done = terminated or truncated
        total_reward += current_reward
        state_action = make_state_action(last_state, last_action, env_name)
        full_obs_data.append(state_action)
        # how much time passed since first frame
        delta = (time.now() - start_time).total_seconds()
        time_data = np.append(time_data, delta)
        if (not run_continuously) and (current_agent_index in wait_agents):
            can_go = False
        while not can_go:
            tm.sleep(1)

    if env_name == TTT_MODE:
        env.show_result(human_render, total_reward)

    # stop the keyboard listener
    listener.stop()
    listener.join()

    output_dict = { "states": state_action_history, "feedback": feedback_history }
    return output_dict

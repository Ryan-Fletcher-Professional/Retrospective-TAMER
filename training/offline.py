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
from networks.Network import Net
from environment.gym_tictactoe.gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark
from environment.snake_gym_custom.snake_gym_custom.envs import SnakeEnvCustom
from environment.GLOBALS import *
from environment.SnakeWrapper import SnakeWrapper
from environment.TTTWrapper import TTTWrapper
from scipy.stats import gamma
from datetime import datetime as time
import time as tm
from environment.MountainCarWrapper import MountainCarWrapper
from training.format_data import make_state_action, make_training_data, make_training_data_with_gamma
from training.train_network import train_network

device = torch.device('cpu')


def offline_collect_feedback(net, env_name, action_history, frame_limit=200, snake_max_fps=20, human_render=True, env=None):
    input_size = MODE_INPUT_SIZES[env_name]
    input_tensor_accumulated = torch.empty((0, input_size), dtype=torch.float32)
    output_tensor_accumulated = torch.empty((0, 2), dtype=torch.float32)

    last_action = 0 # the last action taken
    last_state = np.array([]) # the last state before an action was taken
    state_action_history = np.array([]) # a history of state,action every time feedback was given
    feedback_history = np.array([]) # a history of binary feedbacks
    time_data = np.array([]) # the timestamp of every action starting at 0
    full_obs_data = [] # every state action data ever, even if feedback not provided
    can_go = True
    action_ind = 0

    def on_press(key):
        nonlocal state_action_history, feedback_history, can_go
        nonlocal output_tensor_accumulated, input_tensor_accumulated
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
            output_tensor_accumulated = torch.cat((output_tensor_accumulated, output_tensor), 0)
            input_tensor_accumulated = torch.cat((input_tensor_accumulated, input_tensor), 0)
        else:
            input_tensor, output_tensor = \
                make_training_data(state_action, feedback)
            output_tensor_accumulated = torch.cat((output_tensor_accumulated, output_tensor), 0)
            input_tensor_accumulated = torch.cat((input_tensor_accumulated, input_tensor), 0)
            pass

        can_go = True

    # start a keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    run_continuously = True

    total_reward = 0
    last_state, _ = env.reset()  # This reset will automatically set the start state to be the same as the first time
    # play the game until terminated
    done = False
    start_time = time.now()

    agents = ["random", "env"]
    wait_agents = [1]
    current_agent_index = 0

    while not done:
        # take the next action from action_history
        current_agent_index = (current_agent_index + 1) % len(agents)  # len(agents) = 1 when in single-agent game
        last_action = int(action_history[action_ind])
        action_ind += 1
        # print("action", last_action)
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
    print(input_tensor_accumulated, output_tensor_accumulated)
    return input_tensor_accumulated, output_tensor_accumulated


def offline_no_feedback_run(net, env_name, frame_limit=200, snake_max_fps=20, human_render=True):
    '''
    Use the network to predict actions but do not use any listeners for feedback.
    No training takes place during this run. Only we thing we need to keep track of
    is what actions are taken.
    '''

    last_action = 0
    action_history = np.array([])
    can_go = True

    def get_invalid_actions(inputs):
        invalids = np.zeros(ACTION_SIZES[env_name])
        if env_name == TTT_MODE:
            for j in range(len(invalids)):
                invalids[j] = 1 if inputs[j * 3] == 0 else 0  # First logit for each square is the 'empty square' logit
        if env_name == SNAKE_MODE:
            idx = env.get_invalid_move()
            # print(idx)
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
        env = SnakeWrapper(gym.make(env_name), 'human' if human_render else None, frame_limit=frame_limit, max_fps=snake_max_fps)  # Snake game does not use env.render() so we can't make it not render
    elif (env_name == "snake-v0") or (env_name == "snake-tiled-v0"):
        print("!!!!!!!!\tError: Do you mean to play snake-custom-v0?\t!!!!!!!!")
        return
    else:
        print("!!!!!!!!\tError: No valid environment name\t!!!!!!!!")
        return

    total_reward = 0
    last_state, _ = env.reset()
    # play the game until terminated
    done = False
    while not done:
        # take the action that the network assigns the highest logit value to
        # Note that first we convert from numpy to tensor and then we get the value of the
        # argmax using .item() and feed that into the environment
        current_agent_index = (current_agent_index + 1) % len(agents)  # len(agents) = 1 when in single-agent game
        last_action = agents[current_agent_index](last_state)
        last_state, current_reward, terminated, truncated, info = env.step(last_action)
        done = terminated or truncated
        total_reward += current_reward
        action_history = np.append(action_history, last_action)
        if (not run_continuously) and (current_agent_index in wait_agents):
            # wait a second so people have time to process
            tm.sleep(2)

    if env_name == TTT_MODE:
        env.show_result(human_render, total_reward)

    return action_history, env

def offline_wrapper(net, env_name, frame_limit=200, snake_max_fps=20, human_render=True):
    for i in range(5):
        print("running with no feedback")
        action_history, env = offline_no_feedback_run(net, env_name, frame_limit, snake_max_fps, human_render)
        print("running again, please give feedback")
        indata, outdata = offline_collect_feedback(net, env_name, action_history, frame_limit, snake_max_fps, human_render, env=env)
        train_network(net, indata, outdata)

import random
import sys
import time

import gym
import argparse
import pygame
import torch
from torch.optim import Adam
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from pynput import keyboard
import snake_gym
from gym_tictactoe.gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark
from main.GLOBALS import *
from main.SnakeWrapper import SnakeWrapper
from main.TTTWrapper import TTTWrapper
from networks.Network import Net
from networks.RewardNetwork import *
from scipy.stats import gamma
from datetime import datetime as time
import time as tm

device = torch.device('cpu')


def train_network(net, input, output):
    optimizer = Adam(net.parameters(), lr=0.2)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    #run action,state through policy to get predicted logits for classifying action
    pred_action_logits = net.predict(torch.as_tensor(input, dtype=torch.float32))

    # one hot encode output and make it tensor
    output_one_hot = np.zeros(len(pred_action_logits))
    output_one_hot[output] = 1
    output_tensor = torch.as_tensor(output_one_hot)
    #now compute loss
    loss = loss_criterion(pred_action_logits, output_tensor)
    #back propagate the error through the network
    loss.backward()
    #perform update on policy parameters
    optimizer.step()
    print("Updating network: input ", input, " output: ", output_one_hot)


def train_network_w_gamma(net, state_action_data, feedback, time_data, feedback_delta):
    '''
    net - the network to train
    state_action_data - every single state_action pair from start of game
                        to until feedback was given
    feedback - binary feedback given
    time_data - timestamps (in seconds) for every (state, action)
    feedback_delta - timestamp of the feedback recorded
    '''
    alpha = MOUNTAIN_CAR_GAMMA['alpha']
    loc = MOUNTAIN_CAR_GAMMA['loc']
    scale = MOUNTAIN_CAR_GAMMA['scale']

    state_action_data = np.asarray(state_action_data)

    # copying time_data so I don't modify the original that keeps track
    # of time for each action
    # reverse the time so feedback happens at time 0, and each observation
    # corresponds to how much time has passed from that observation until
    # feedback was recorded
    time_data_mod = np.copy(time_data)
    time_data_mod = np.append(time_data_mod, feedback_delta)
    time_data_mod = time_data_mod[-1]-np.asarray(time_data_mod)
    n = len(time_data_mod)

    # use gamma function to decide on how feedback credits should be
    # assigned
    credits = gamma.cdf(time_data_mod[0: n-1], alpha, loc, scale) \
            - gamma.cdf(time_data_mod[1: n], alpha, loc, scale)

    # for an almost 0 credit, better to not pass it to network at all
    # find the index of that credit cutoff
    credits_cutoff_ind = np.argmax(credits>GAMMA_CREDIT_CUTOFF)
    credits = credits[credits_cutoff_ind:]
    state_action_data = state_action_data[credits_cutoff_ind:]
    # convert this where each credit is [0, credit] or [credit, 0] depending
    # on feedback
    network_output = np.zeros((len(credits), 2))
    network_output[:, feedback] = credits


    input_tensor = torch.as_tensor(state_action_data, dtype=torch.float32).reshape(state_action_data.shape[0], -1)
    output_tensor = torch.as_tensor(network_output).reshape(network_output.shape[0], -1)


    # predict for all observations
    pred_action_logits = net(input_tensor)

    # optimizer
    optimizer = Adam(net.parameters(), lr=0.2)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()

    #now compute loss
    loss = loss_criterion(pred_action_logits, output_tensor)
    #back propagate the error through the network
    loss.backward()
    #perform update on policy parameters
    optimizer.step()
    print("Trained based on feedback", feedback)


# def train_network(net, input, output):
#     optimizer = Adam(net.parameters(), lr=0.2)
#     loss_criterion = nn.CrossEntropyLoss()
#     optimizer.zero_grad()
#
#     #run action,state through policy to get predicted logits for classifying action
#     pred_action_logits = net(torch.as_tensor(input, dtype=torch.float32))
#     # one hot encode output and make it tensor
#     output_one_hot = np.zeros(2)
#     output_one_hot[output] = 1
#     output_tensor = torch.as_tensor(output_one_hot)
#     #now compute loss
#     loss = loss_criterion(pred_action_logits, output_tensor)
#     #back propagate the error through the network
#     loss.backward()
#     #perform update on policy parameters
#     optimizer.step()


def make_state_action(state, action, n_actions=3):
    '''
    Takes in the state as obs np array
    and the action as an integer
    returns np array of state and one-hot encoded action
    '''
    # one hot encode the actions
    one_hot_action = np.zeros(n_actions)
    one_hot_action[action] = 1
    # append the state, actions to obs_data
    state_action = np.append(state, one_hot_action)

    return state_action



# currently only works with mountain car
def collect_live_data(net, env_name, human_render=True):
    '''
    Run a live simulation and collect keyboard data
    inputs: policy, gym environment name, and whether to render
    outputs: a list of dictionaries {"state": np.array, "feedback": binary}
    '''
    # feedback_data = []
    # obs_data = []
    # time_data = []
    
    

    last_action = 0
    last_state = np.array([])
    state_action_history = np.array([])
    feedback_history = np.array([])
    time_data = np.array([])
    full_obs_data = np.array([])
    can_go = True

    def on_press(key):
        nonlocal state_action_history, feedback_history, can_go
        if(len(full_obs_data)==0):
            print("Slow down! Haven't even started playing yet.")
            return
        try:
            # c for negative feedback
            if key.char=='c':
                feedback=0
            # v for positive feedback
            elif key.char=='v':
                feedback=1
            else:
                print("WRONG KEY! Press 'c' or 'v'")
                return
        except Exception as e:
            print("WRONG INPUT! Press 'c' or 'v'")
            print("Exception: ", e)
            return

        state_action = make_state_action(last_state, last_action)
        state_action_history = np.append(state_action_history, state_action)
        feedback_history = np.append(feedback_history, feedback)
        if env_name == MOUNTAIN_CAR_MODE:
            # how much time passed since first frame
            # and the time the  feedback was recorded
            feedback_delta = (time.now()-start_time).total_seconds()
            train_network_w_gamma(net, full_obs_data, feedback, time_data, feedback_delta)
            # TODO : Train snake w/gamma
        else:
            train_network(net, state_action, feedback)

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
            print("TODO : SNAKE INVALID ACTIONS")  # TODO : Return action logit indicating 'move back the way it came'

        return invalids

    agents = [lambda net_input: net.predict_max_action(net_input, get_invalid_actions(net_input))]
    current_agent_index = 0
    wait_agents = [0]  # Which agents to wait for when not running continuously

    run_continuously = True

    if env_name == TTT_MODE:
        # Wraps the TTT environment to alter arguments for version compatibility
        env = TTTWrapper(TicTacToeEnv(), '0', human_render)
        agents.append(lambda _: random.choice(env.available_actions()))
        run_continuously = False
    elif env_name == SNAKE_MODE:
        env = SnakeWrapper(gym.make(env_name))  # Snake game does not use env.render() so we can't make it not render
    elif human_render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)

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
        state_action = make_state_action(last_state, last_action)
        full_obs_data = np.append(full_obs_data, state_action)
        # how much time passed since first frame
        delta = (time.now()-start_time).total_seconds()
        time_data = np.append(time_data, delta)
        if (not run_continuously) and (current_agent_index in wait_agents):
            can_go = False
        while not can_go:
            time.sleep(1)

    if env_name == TTT_MODE:
        env.show_result(human_render, total_reward)

    # stop the keyboard listener
    listener.stop()
    listener.join()

    output_dict = { "states": state_action_history, "feedback": feedback_history }
    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_plays', default=5, type=int, help="number of play runs to collect")
    #parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    #parser.add_argument('--num_inv_dyn_iters', default = 500, type=int, help="number of iterations to train inverse dynamics model")
    #parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    args = parser.parse_args()
    play_iters = args.num_plays

    mode = SNAKE_MODE
    net = Net(mode)
    for i in range(play_iters):
        data = collect_live_data(net, env_name=mode)
        print("Data for run " + str(i + 1) + ":\n" + str(data))

    #collect human demos

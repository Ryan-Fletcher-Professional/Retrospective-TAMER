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
from main.GLOBALS import *
from networks.RewardNetwork import *
from scipy.stats import gamma
from datetime import datetime as time
import time as tm

device = torch.device('cpu')


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
    one_hot_action[action]=1
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
    feedback_data = []
    obs_data = []
    time_data = []
    full_obs_data = []

    # called when a key is pressed
    def on_press(key):
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
            print("Exception: ", e)
            return

        state_action = make_state_action(obs, action)

        obs_data.append(state_action)
        feedback_data.append(feedback)

        # how much time passed since first frame
        # and the time the  feedback was recorded
        feedback_delta = (time.now()-start_time).total_seconds()

        #train_network(net, state_action, feedback)
        train_network_w_gamma(net, full_obs_data, feedback, time_data, feedback_delta)



    # start a keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    # whether to render the game in a window or not
    if human_render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)


    done = False
    total_reward = 0
    obs, _ = env.reset()
    # default action do nothing
    action = 1

    #state_action = make_state_action(obs, action)
    #full_obs_data.append(state_action)

    # play the game until terminated
    start_time = time.now()
    while not done:
        # take the action that the network assigns the highest logit value to
        action = net.predict_max_action(obs)
        print("action = ", action)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += rew

        state_action = make_state_action(obs, action)
        full_obs_data.append(state_action)

        # how much time passed since first frame
        delta = (time.now()-start_time).total_seconds()
        time_data.append(delta)


    # stop the keyboard listener
    listener.stop()
    listener.join()

    output_dict = {"states":np.asarray(obs_data), "feedback":np.asarray(feedback_data)}
    return output_dict


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_plays', default = 1, type=int, help="number of play runs")
    #parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    #parser.add_argument('--num_inv_dyn_iters', default = 500, type=int, help="number of iterations to train inverse dynamics model")
    #parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    args = parser.parse_args()
    play_iters = args.num_plays

    net = CarRewardNet()
    for i in range(play_iters):
        data = collect_live_data(net, env_name="MountainCar-v0")


    #collect human demos

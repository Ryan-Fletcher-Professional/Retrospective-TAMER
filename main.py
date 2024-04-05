import random
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
from gym_tictactoe.env import TicTacToeEnv, agent_by_mark, check_game_status,\
    after_action_state, tomark, next_mark
from main.GLOBALS import *
from networks.Network import Net
from networks.RewardNetwork import *

device = torch.device('cpu')


def train_network(net, input, output):
    optimizer = Adam(net.parameters(), lr=0.2)
    loss_criterion = nn.CrossEntropyLoss()
    optimizer.zero_grad()
    #run action,state through policy to get predicted logits for classifying action
    pred_action_logits = net(torch.as_tensor(input, dtype=torch.float32))

    # one hot encode output and make it tensor
    output_one_hot = np.zeros(2)
    output_one_hot[output]=1
    output_tensor = torch.as_tensor(output_one_hot)
    #now compute loss
    loss = loss_criterion(pred_action_logits, output_tensor)
    #back propagate the error through the network
    loss.backward()
    #perform update on policy parameters
    optimizer.step()
    print("Updating network: input ", input, " output: ", output_one_hot)


# currently only works with mountain car
def collect_live_data(net, env_name, human_render=True):
    '''
    Run a live simulation and collect keyboard data
    inputs: policy, gym environment name, and whether to render
    outputs: a list of dictionaries {"state": np.array, "feedback": binary}
    '''

    if(mode == TTT_MODE):
        collect_live_data_ttt(net, human_render)
        return

    feedback_data = []
    obs_data = []

    # called when a key is pressed
    def on_press(key):
        # c for negative feedback
        if key.char == 'c':
            feedback = 0
        # v for positive feedback
        elif key.char == 'v':
            feedback = 1
        else:
            print("WRONG KEY! Press 'c' or 'v'")
            return

        # one hot encode the actions
        one_hot_action = np.zeros(3)
        one_hot_action[action] = 1

        # append the state, actions to obs_data
        state_action = np.append(obs, one_hot_action)
        obs_data.append(state_action)
        feedback_data.append(feedback)
        train_network(net, state_action, feedback)

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
    # play the game until terminated
    while not done:

        # take the action that the network assigns the highest logit value to
        # Note that first we convert from numpy to tensor and then we get the value of the
        # argmax using .item() and feed that into the environment
        #action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
        action = net.predict_max_action(obs)
        #print(action)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += rew

    # stop the keyboard listener
    listener.stop()
    listener.join()

    output_dict = {"states": np.asarray(obs_data), "feedback": np.asarray(feedback_data)}
    return output_dict


# currently only works with mountain car
def collect_live_data_ttt(net, human_render=True):
    '''
    Run a live simulation and collect keyboard data
    inputs: policy, gym environment name, and whether to render
    outputs: a list of dictionaries {"state": np.array, "feedback": binary}
    '''
    feedback_data = []
    obs_data = []

    feedback_data = []
    action = 0

    # called when a key is pressed
    def on_press(key):
        # c for negative feedback
        if key.char == 'c':
            feedback = 0
        # v for positive feedback
        elif key.char == 'v':
            feedback = 1
        else:
            print("WRONG KEY! Press 'c' or 'v'")
            return

        # one hot encode the actions
        one_hot_action = np.zeros(3)
        one_hot_action[action] = 1

        # append the state, actions to obs_data
        state_action = np.append(obs, one_hot_action)
        obs_data.append(state_action)
        feedback_data.append(feedback)
        train_network(net, state_action, feedback)

    # start a keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    total_reward = 0
    start_mark = 'O'
    env = TicTacToeEnv()
    agents = [lambda _, actions: random.choice(actions),
              lambda obs, _: net.predict_max_action(obs)]

    env.set_start_mark(start_mark)
    obs = env.reset()
    i = 0
    while not env.done:
        _, mark = obs
        env.show_turn(human_render, mark)
        agent = agents[i]
        i = (i + 1) % len(agents)
        ava_actions = env.available_actions()
        action = agent(obs, ava_actions)
        obs, reward, done, info = env.step(action)
        total_reward += reward
        if human_render:
            env.render()

    env.show_result(human_render, mark, reward)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_plays', default=1, type=int, help="number of play runs to collect")
    #parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    #parser.add_argument('--num_inv_dyn_iters', default = 500, type=int, help="number of iterations to train inverse dynamics model")
    #parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    args = parser.parse_args()
    play_iters = args.num_plays

    # mode = TTT_MODE
    # net = Net(mode)
    # data = collect_live_data(net, env_name=mode)
    # print(data)

    mode = TTT_MODE
    net = Net(mode)
    for i in range(play_iters):
        data = collect_live_data(net, env_name=mode)
        print("Data for run " + str(i + 1) + ":\n" + str(data))

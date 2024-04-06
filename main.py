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
from main.TTTWrapper import TTTWrapper
from networks.Network import Net
from networks.RewardNetwork import *

device = torch.device('cpu')


def collect_live_data(net: Net, env_name: str, human_render: bool = True):
    """
        Run a live simulation and collect keyboard data
        inputs: policy, gym environment name, and whether to render
        outputs: a list of dictionaries {"state": np.array, "feedback": binary}
    """

    last_action = 0
    last_state = np.array([])
    state_action_history = np.array([])
    feedback_history = np.array([])
    can_go = True

    def on_press(key):
        nonlocal state_action_history, feedback_history, can_go
        # c for negative feedback
        if key.char == 'c':
            feedback = 0
        # v for positive feedback
        elif key.char == 'v':
            feedback = 1
        else:
            print("WRONG KEY! Press 'c' or 'v'")

        # one hot encode the actions
        one_hot_action = np.zeros(ACTION_SIZES[env_name])
        one_hot_action[last_action] = 1

        # append the state, actions to obs_data
        state_action = np.append(last_state, one_hot_action)
        state_action_history = np.append(state_action_history, state_action)
        feedback_history = np.append(feedback_history, feedback)
        train_network(net, state_action, feedback)

        can_go = True

    listener = keyboard.Listener(on_press=on_press)
    listener.start()

    def get_invalid_actions(inputs):
        invalids = np.zeros(ACTION_SIZES[env_name])
        if env_name == MOUNTAIN_CAR_MODE:
            return np.zeros(MOUNTAIN_CAR_ACTION_SIZE)
        if env_name == TTT_MODE:
            for j in range(len(invalids)):
                invalids[j] = 1 if inputs[j * 3] == 0 else 0  # First logit for each square is the 'empty square' logit
            return invalids
        if env_name == SNAKE_MODE:
            pass  # TODO : Return action logit indicating 'move back the way it came'

    agents = [lambda net_input: net.predict_max_action(net_input, get_invalid_actions(net_input))]
    current_agent_index = 0
    wait_agents = [0]  # Which agents to wait for when not running continuously

    run_continuously = True

    if env_name == TTT_MODE:
        # Wraps the TTT environment to alter arguments for version compatibility
        env = TTTWrapper(TicTacToeEnv(), '0', human_render)
        agents.append(lambda _: random.choice(env.available_actions()))
        run_continuously = False
    elif human_render:
        env = gym.make(env_name, render_mode='human')
    else:
        env = gym.make(env_name)

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
        # print(action)
        last_state, current_reward, terminated, truncated, info = env.step(last_action)
        if terminated or truncated:
            done = True
        total_reward += current_reward
        if (not run_continuously) and (current_agent_index in wait_agents):
            can_go = False
        while not can_go:
            time.sleep(1)

    if env_name == TTT_MODE:
        env.show_result(human_render, total_reward)

    # stop the keyboard listener
    listener.stop()
    listener.join()

    output_dict = {"states": np.asarray(state_action_history), "feedback": np.asarray(feedback_history)}
    return output_dict



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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=None)
    parser.add_argument('--num_plays', default=1, type=int, help="number of play runs to collect")
    #parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    #parser.add_argument('--num_inv_dyn_iters', default = 500, type=int, help="number of iterations to train inverse dynamics model")
    #parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")
    args = parser.parse_args()
    play_iters = args.num_plays

    mode = MOUNTAIN_CAR_MODE
    net = Net(mode)
    for i in range(play_iters):
        data = collect_live_data(net, env_name=mode)
        print("Data for run " + str(i + 1) + ":\n" + str(data))

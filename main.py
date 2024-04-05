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


device = torch.device('cpu')


class PolicyNetwork(nn.Module):
    '''
        Simple neural network with two layers that maps a 2-d state to a prediction
        over which of the three discrete actions should be taken.
        The three outputs corresponding to the logits for a 3-way classification problem.

    '''
    def __init__(self):
        super().__init__()

        #This layer has 2 inputs corresponding to car position and velocity
        self.fc1 = nn.Linear(2, 8)
        #This layer has three outputs corresponding to each of the three discrete actions
        self.fc2 = nn.Linear(8, 3)

    def forward(self, x):
        #this method performs a forward pass through the network, applying a non-linearity (ReLU) on the
        #outputs of the first layer
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# currently only works with mountain car
def collect_live_data(pi, env_name, human_render=True):
    '''
    Run a live simulation and collect keyboard data
    inputs: policy, gym environment name, and whether to render
    outputs: a list of dictionaries {"state": np.array, "feedback": binary}
    '''
    feedback_data = []
    # called when a key is pressed
    def on_press(key):
        # c for negative feedback
        if key.char=='c':
            feedback_data.append({"state": obs, "feedback": 0})
        # v for positive feedback
        if key.char=='v':
            feedback_data.append({"obs": obs, "feedback": 1})

    # start a keyboard listener
    listener = keyboard.Listener(on_press=on_press)
    listener.start()

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
        action = torch.argmax(pi(torch.from_numpy(obs).unsqueeze(0))).item()
        #print(action)
        obs, rew, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        total_reward += rew

    # stop the keyboard listener
    listener.stop()
    listener.join()
    return feedback_data


if __name__ == "__main__":
    #parser = argparse.ArgumentParser(description=None)
    #parser.add_argument('--num_demos', default = 1, type=int, help="number of human demonstrations to collect")
    #parser.add_argument('--num_bc_iters', default = 100, type=int, help="number of iterations to run BC")
    #parser.add_argument('--num_inv_dyn_iters', default = 500, type=int, help="number of iterations to train inverse dynamics model")
    #parser.add_argument('--num_evals', default=6, type=int, help="number of times to run policy after training for evaluation")

    #args = parser.parse_args()
    pi = PolicyNetwork()
    data = collect_live_data(pi, env_name="MountainCar-v0")
    print(data)

    #collect human demos

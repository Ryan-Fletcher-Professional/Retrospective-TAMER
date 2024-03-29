import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

from main.GLOBALS import *


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    # Build a feedforward neural network for the policy
    layers = []
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self, mode=DEFAULT_MODE, sizes=None):
        super().__init__()
        if sizes is None:
            sizes = DEFAULT_POLICY_SIZES[mode]
        self.network = mlp(sizes)

    def predict_return(self, traj):
        # calculate return (cumulative reward) of a trajectory (could be any number of timesteps)
        return self.network(traj).sum()

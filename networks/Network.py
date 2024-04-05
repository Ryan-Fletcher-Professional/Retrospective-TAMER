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
    def __init__(self, mode: str = DEFAULT_MODE, sizes: [int] = None):
        super().__init__()
        self.mode = mode
        if sizes is None:
            sizes = DEFAULT_POLICY_SIZES[mode]
        self.network = mlp(sizes)

    def predict(self, state_action):
        out = self.network(state_action)
        return out

    def predict_max_action(self, state):
        # creates a matrix [1, 0, 0,
        #                   0, 1, 0,
        #                   0, 0, 1]

        action_space = np.eye(3)
        max_action_ind = 0
        preds = []
        print("--------------- " + self.mode + " ---------------")
        for action in action_space:
            state_action = torch.as_tensor(np.append(state, action), dtype=torch.float32)
            new_pred = self.predict(state_action)
            new_pred_proba = nn.functional.softmax(new_pred)
            preds.append(new_pred_proba[1].item())
            print(new_pred_proba)
        best_action_ind = np.argmax(np.asarray(preds))
        print("best action", best_action_ind)
        return best_action_ind

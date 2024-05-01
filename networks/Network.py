import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np

from environment.GLOBALS import *


def mlp(sizes, activation=nn.ReLU, output_activation=nn.Identity):
    # Build a feedforward neural network for the policy
    layers = []
    #print(sizes)
    for j in range(len(sizes) - 1):
        act = activation if j < len(sizes) - 2 else output_activation
        layers += [nn.Linear(sizes[j], sizes[j + 1]), act(sizes[j + 1]) if act == nn.LayerNorm else act()]
    return nn.Sequential(*layers)


class Net(nn.Module):
    def __init__(self, mode: str = DEFAULT_MODE, sizes: [int] = None):
        super().__init__()
        self.mode = mode
        if sizes is None:
            sizes = DEFAULT_POLICY_SIZES[mode]
        self.action_size = ACTION_SIZES[mode]
        self.network = mlp(sizes)

    def predict(self, state_action):
        #print(state_action)
        out = self.network(state_action)
        out = nn.functional.softmax(out, dim=0)
        return out

    def predict_max_action(self, state, invalid_actions):
        # creates a logit matrix [ 1, 0, 0,
        #                          0, 1, 0,
        #                          0, 0, 1...
        #                            ...      ]
        action_space = np.eye(self.action_size)
        preds = []
        title = "--------------- " + self.mode + ".predict_max_action ---------------"
        line = "-"*len(title)
        #print(line + "\n" + title + line)
        for i in range(self.action_size):
            if invalid_actions[i] == 1:
                preds.append(0)
            else:
                #print("STATE: " + str(state))
                #print("ACTION: " + str(action_space[i]), "action size", self.action_size)
                if self.mode == DEFAULT_MODE:
                    action_space[i][1] = 1*(state[1]>0)

                state_action = torch.as_tensor(np.append(state, action_space[i]), dtype=torch.float32)

                new_pred = self.predict(state_action)
                print("Actual predictions:", new_pred)
                # print(new_pred)
                new_pred_proba = nn.functional.softmax(new_pred, dim=0)
                preds.append(new_pred_proba[1].item())
        # print("PREDICTIONS:", preds)
        best_action_ind = np.argmax(np.asarray(preds))
        #print(line + "\n")
        return best_action_ind

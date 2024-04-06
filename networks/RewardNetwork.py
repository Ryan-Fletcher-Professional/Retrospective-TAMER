import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.optim import Adam
import numpy as np


'''
# TODO
Ideally we should only make one network class that takes in dimensions
as arguments and instantiate it for different games. Right now goal is to make
it work so I'm making one specific for mountain car
'''
class CarRewardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 2)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict(self, state_action):
        out = self(state_action)
        return out

    def predict_max_action(self, state):
        # creates a matrix [1, 0, 0,
        #                   0, 1, 0,
        #                   0, 0, 1]

        action_space = np.eye(3)
        max_action_ind = 0
        preds = []
        #print("----------------------")
        for action in action_space:
            state_action = torch.as_tensor(np.append(state, action), dtype=torch.float32)
            new_pred = self.predict(state_action)
            new_pred_proba = nn.functional.softmax(new_pred)
            preds.append(new_pred_proba[1].item())
            #print(new_pred_proba)
        best_action_ind = np.argmax(np.asarray(preds))
        #print("predicted action", best_action_ind)
        return best_action_ind

import Network
import torch.nn as nn


class RewardNetwork(Network):
    def __init__(self, sizes=None):
        super().__init__(sizes)


'''
# TODO
Ideally we should only make one network class that takes in dimensions
as arguments and instantiate it for different games. Right now goal is to make
it work so I'm making one specific for mountain car
'''
class CarRewardNet(nn.Module):

    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(2, 16)
        self.fc2 = nn.Linear(16, 16)
        self.fc3 = nn.Linear(16, 3)

    def forward(self, x):
        x = F.leaky_relu(self.fc1(x))
        x = F.leaky_relu(self.fc2(x))
        x = self.fc3(x)
        return x

    def predict_return(self, traj):
        out = self(traj)
        return out

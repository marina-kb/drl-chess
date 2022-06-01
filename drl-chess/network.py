""""
Neural Network
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


# Version Paul

class Net(nn.Module):
    """
    CNN DQN
    """

    def __init__(self):
        super(Net, self).__init__()

        self.net = nn.Sequential(
          nn.Conv2d(8, 111, 8, 8),
          nn.ReLU(),
        #   nn.Conv2d(10, 25, 3, 1),
        #   nn.ReLU(),
          nn.Linear(4672,1),
          )

    def forward(self, x):
        return self.net(x)


class DQN(nn.Module):
    """
    A simple Deep Q-Network with 3 linear layers.
    """

    def __init__(self):
        super().__init__()

        self.net = nn.Sequential(
            torch.flatten(obs_new),
            nn.Linear(1, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 4672),
            nn.ReLU(inplace=True),
        )

    def forward(self, obs):
        return self.net(obs)

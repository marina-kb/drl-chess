""""
Neural Network
"""
import torch
import torch.nn as nn


class Conv(nn.Module):
    """
    CNN DQN
    """

    def __init__(self):
        super(Conv, self).__init__()

        self.net = nn.Sequential(
            nn.Conv2d(111, 64, kernel_size=2, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, kernel_size=2, stride=1, padding=1),
            #nn.Conv2d(64, 32, 3, 3),
            nn.ReLU(),
            nn.Conv2d(32, 8, kernel_size=2, stride=1, padding=1),
            nn.ReLU()
            # nn.ReLU(),
            # nn.Linear(1443, 1),
            # nn.Linear(4672,1),
        )

        self.linear = nn.Sequential(
            nn.Linear(968, 2048),
            nn.Linear(2048, 4672),
        )

    def forward(self, x):
        y = self.net(x)
        # print(y.shape)
        y = torch.flatten(y, start_dim=1, end_dim=-1)
        # print(y.shape)

        return self.linear(y)


class Linear(nn.Module):
    """
    A simple Deep Q-Network with 3 linear layers.
    """

    def __init__(self):
        super(Linear, self).__init__()

        self.net = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(7104, 264),    # arg1 = 111 * 8 * 8 parameters
            # nn.ReLU(inplace=True),
            nn.Linear(264, 264),
            nn.Dropout(p=0.2),
            nn.Linear(264, 128),
            nn.Linear(128, 4672),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        y = self.net(x)
        # y = torch.flatten(y, start_dim=0, end_dim=-2)
        return y

""""
Neural Network
"""
import torch
import torch.nn as nn

from config import CFG


class Conv(nn.Module):
    """
    CNN DQN
    """

    def __init__(self):
        super(Conv, self).__init__()
        self.input_shape = 20 if CFG.small_obs else 111

        self.net = nn.Sequential(
            nn.Conv2d(self.input_shape, 64, kernel_size=2, stride=1, padding=1),
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
        self.input_shape = 20 if CFG.small_obs else 111

        self.net = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear((self.input_shape * 64), 264),    # arg1 = 111 * 8 * 8 parameters
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(264),
            nn.Linear(264, 264),
            # nn.BatchNorm1d(264),
            nn.Linear(264, 128),
            # nn.BatchNorm1d(128),
            nn.Linear(128, 4672),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        y = self.net(x)
        # y = torch.flatten(y, start_dim=0, end_dim=-2)
        return y


class DistinctLayer(nn.Module):
    """
    A multi layer network that distinguishes 'global' and 'piece-centric' layers.
    Use only with old version layering (CFG.twnty_obs = True).
    """

    def __init__(self):
        super(Linear, self).__init__()

        self.glob = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear((7 * 64), 128),
            nn.ReLU(inplace=True)
        )

        self.pieces = nn.Sequential(

            nn.ReLU(inplace=True)
        )

        self.reunion = nn.Sequential(

            nn.ReLU(inplace=True)
        )

    def forward(self, x):

        y_g = self.glob(x[:, 0:7])

        y_p = self.pieces(x[:, 7:21])

        # x =  y_g.extend(y_p) ==> Torch Cat

        y = self.reunion(x)

        return y

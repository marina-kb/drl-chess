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
            nn.Conv2d(self.input_shape, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(),
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.LeakyReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(800, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4672),
            nn.Tanh()
        )

    def forward(self, x):
        y = self.net(x)

        y = torch.flatten(y, start_dim=1, end_dim=-1)

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
            nn.Linear((self.input_shape * 64), 256),    # arg1 = 111 * 8 * 8 parameters
            # nn.ReLU(inplace=True),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 256),
            # nn.BatchNorm1d(256),
            nn.Linear(256, 128),
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
        super(DistinctLayer, self).__init__()

        self.glob = nn.Sequential(
            nn.Conv2d(7, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.pieces = nn.Sequential(
            nn.Conv2d(13, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(),
            nn.Conv2d(64, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )

        self.net = nn.Sequential(
            nn.Conv2d(64, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(),
            nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(),
            nn.Conv2d(128, 32, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(),
        )


        self.linear = nn.Sequential(
            nn.Linear(2048, 2048),
            nn.LeakyReLU(),
            nn.Linear(2048, 4672),
            nn.Tanh()
        )

    def forward(self, x):

        y_g = self.glob(x[:, 0:7, :, :])
        # print(y_g.shape)

        y_p = self.pieces(x[:, 7:21, :, :])

        y = torch.cat((y_g, y_p), 1)
        # print(y.shape)

        y = self.net(y)
        # print(y.shape)

        y = torch.flatten(y, start_dim=1, end_dim=-1)

        return self.linear(y)

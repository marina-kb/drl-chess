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
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            #nn.MaxPool2d(3, stride=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 16, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(),
            nn.Conv2d(16, 8, kernel_size=1, stride=1, padding=1),
            nn.BatchNorm2d(8),
            nn.ReLU()
        )

        self.linear = nn.Sequential(
            nn.Linear(800, 2048),
            nn.ReLU(),
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
            nn.Conv2d(7, 7, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.pieces = nn.Sequential(
            nn.Conv2d(13, 13, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )

        self.reunion = nn.Sequential(
            nn.Flatten(1, -1),
            nn.Linear(1280, 1600),
            nn.Linear(1600, 4672),
        )

    def forward(self, x):

        y_g = self.glob(x[:, 0:7, :, :])
        # y_g = torch.flatten(y_g, start_dim=1, end_dim=-1)

        y_p = self.pieces(x[:, 7:21, :, :])
        # y_p = torch.flatten(y_p, start_dim=1, end_dim=-1)

        y = torch.cat((y_g, y_p), 1)

        # print(y.shape)

        return self.reunion(x)



# def __init__(self):
#         super(DistinctLayer, self).__init__()

#         self.glob = nn.Sequential(
#             nn.Flatten(1, -1),
#             nn.Linear((7 * 64), 128),
#             nn.ReLU(inplace=True)
#         )

#         self.pieces = nn.Sequential(
#             nn.Flatten(1, -1),
#             nn.Linear((13 * 64), 128),
#             nn.ReLU(inplace=True)
#         )

#         self.reunion = nn.Sequential(
#             nn.Linear(256, 128),
#             nn.Linear(128, 4672),
#             nn.ReLU(inplace=True)
#         )

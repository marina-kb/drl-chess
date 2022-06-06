"""
Agent module.
"""

from code import interact
import random
import numpy as np
import chess.engine
from pettingzoo.classic.chess import chess_utils
import torch
import collections

from data import DAT
from config import CFG
from engine import Engine
from network import Conv, Linear


class Agent:
    def __init__(self):
        pass

    def move(self, observation, board):
        pass

    def feed(self, obs_old, act, rwd, obs_new):
        pass


class DeepK(Agent):
    def __init__(self):
        super().__init__()

        agt_t = {"conv": Conv, "linear": Linear}

        self.net = agt_t[CFG.net_type]()
        self.obs = collections.deque(maxlen=CFG.buffer_size)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)

    def feed(self, obs_old, act, rwd, obs_new):
        """
        Learn from a single observation sample.
        """

        old = torch.tensor(obs_old["observation"])
        act = torch.tensor(act)
        rwd = torch.tensor(rwd)
        new = torch.tensor(obs_new["observation"])

        self.obs.append((old, act, rwd, new))
        if len(self.obs) >= CFG.batch_size:
            self.learn()

    def learn(self):

        batch = random.sample(self.obs, CFG.batch_size)
        old, act, rwd, new = zip(*batch)

        old = torch.stack(old).type(torch.FloatTensor)
        act = torch.stack(act).long().unsqueeze(1)
        rwd = torch.stack(rwd).type(torch.FloatTensor)
        new = torch.stack(new).type(torch.FloatTensor)

        # We get the network output
        out = torch.gather(self.net(old), 1, act).squeeze(1)

        # We compute the target
        with torch.no_grad():
            idx = torch.argmax(self.net(new), 1).unsqueeze(1)
            exp = rwd + CFG.gamma * torch.gather(self.net(new), 1, idx).squeeze(1)

        # Compute the loss
        loss = torch.square(exp - out)
        if CFG.debug:
            print("loss", loss, "\n")
        DAT.set_loss(loss.tolist())

        # Perform a backward propagation.
        self.opt.zero_grad()
        loss.sum().backward()
        self.opt.step()

        # stats_rwd(rwd): TODO later

    def move(self, new_obs, board):
        """
        Run an epsilon-greedy policy for next action selection.
        """

        # Return random action with probability epsilon
        if random.uniform(0, 1) < CFG.epsilon or len(self.obs) <= CFG.batch_size:
            if CFG.debug:
                print("Deep_K Epsilon-induced Random Move !")
            return random.choice(np.flatnonzero(new_obs["action_mask"]))

        # Else, return action with highest value

        with torch.no_grad():
            new = torch.tensor(new_obs["observation"]).type(torch.FloatTensor).unsqueeze(0)
            val = self.net(new).squeeze(0)
            valid_actions = torch.tensor(
                np.squeeze(np.argwhere(new_obs["action_mask"] == 1).T, axis=0)
            )
            valid_q = torch.index_select(val, 0, valid_actions)
            best_q = torch.argmax(valid_q).numpy()
            action = valid_actions[best_q].numpy()
        return action.tolist()


class Random(Agent):
    def __init__(self):
        super().__init__()

    def move(self, observation, board):
        return random.choice(np.flatnonzero(observation["action_mask"]))


class StockFish(Agent):
    def __init__(self):
        super().__init__()
        if CFG.engine == None:
            CFG.engine = Engine()

    @staticmethod
    def move_to_act(move):
        """
        Convert a string-like move to an action.
        """
        x, y = chess_utils.square_to_coord(move.from_square)
        panel = chess_utils.get_move_plane(move)
        return (x * 8 + y) * 73 + panel

    def _move(self, obs, board):
        if board.turn == False:
            board = board.mirror()
        move = CFG.engine.engine.play(board=board, limit=CFG.engine.limit)
        return self.move_to_act(move.move)

    def move(self, observation, board):
        move = self._move(observation, board)
        return move


class ObservationGenerator(StockFish, DeepK):
    def __init__(self):
        super().__init__()

    def move(self, new_obs, board):
        return super()._move(new_obs, board)

    def learn(self):
        return

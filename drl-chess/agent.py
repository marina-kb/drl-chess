"""
Agent module.
"""

import random
import numpy as np
import chess.engine
from pettingzoo.classic.chess import chess_utils
import torch
import collections

from config import CFG
from engine import Engine
import network

class Agent():
    def __init__(self):
        pass

    def move(self, observation, board):
        pass

    def feed(self, obs_old, act, rwd, obs_new):
        pass


class DeepKasp_Lin(Agent):
    def __init__(self):
        super().__init__()
        self.net = network.Lin()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)


    def feed(self, obs_old, act, rwd, obs_new):
        """
        Learn from a single observation sample.
        """

        obs_old = torch.tensor(obs_old['observation'])
        obs_new = torch.tensor(obs_new['observation'])

        # We get the network output
        out = self.net(torch.tensor(obs_new))[act]

        # We compute the target
        with torch.no_grad():
            exp = rwd + CFG.gamma * self.net(obs_new).max()

        # Compute the loss
        loss = torch.square(exp - out)

        # Perform a backward propagation.
        self.opt.zero_grad()
        loss.sum().backward()
        self.opt.step()


    def move(self, new_obs, board):
        """
        Run an epsilon-greedy policy for next actino selection.
        """
        # Return random action with probability epsilon
        if random.uniform(0, 1) < CFG.epsilon:
            return board.sample()
        # Else, return action with highest value
        with torch.no_grad():
            val = self.net(torch.tensor(new_obs))
            return torch.argmax(val).numpy()


class DeepKasp_Conv(Agent):
    def __init__(self):
        super().__init__()
        if CFG.agt_type == "Conv":
            self.net = network.Conv()
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_list = []


    def feed(self, obs_old, act, rwd, obs_new):
        """
        Learn from a single observation sample.
        """
        obs_old = torch.tensor(obs_old['observation'])
        obs_new = torch.tensor(obs_new['observation'])

        # We get the network output
        out = self.net(obs_new.type(torch.FloatTensor))[act]

        # We compute the target
        with torch.no_grad():
            exp = rwd + CFG.gamma * self.net(obs_new.type(torch.FloatTensor)).max()

        # Compute the loss
        loss = torch.square(exp - out)
        self.loss_list.append(loss.tolist())

        # Perform a backward propagation.
        self.opt.zero_grad()
        loss.sum().backward()
        self.opt.step()


    def move(self, new_obs, board):
        """
        Run an epsilon-greedy policy for next actino selection.
        """
        # Return random action with probability epsilon
        if random.uniform(0, 1) < CFG.epsilon:
            if CFG.debug:
                print("Deep_K Epsilon-induced Random Move !")
            return random.choice(np.flatnonzero(new_obs["action_mask"]))

        # Else, return action with highest value
        with torch.no_grad():
            # to clean up & squeeze ?..
            val = self.net(torch.tensor(new_obs['observation']).type(torch.FloatTensor))
            valid_actions = torch.tensor(np.squeeze(np.argwhere(new_obs['action_mask'] == 1).T, axis=0))
            valid_q = torch.index_select(val, 0, valid_actions)
            best_q = torch.argmax(valid_q).numpy()
            action = valid_actions[best_q].numpy()
        if CFG.debug:
            print("DeepK Engine Move: ", action.tolist())
        return action.tolist()


# new agent with batches (experience replay)
class DeepKasp_Conv_Batch(Agent):
    def __init__(self):
        super().__init__()
        self.net = network.Conv()
        #to add batch_size 32 & maxlen 100 in config ?
        self.obs = collections.deque(maxlen=10000)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=0.0001)
        self.loss_list = []


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
        act = torch.stack(act).type(torch.FloatTensor)
        rwd = torch.stack(rwd).type(torch.FloatTensor)
        new = torch.stack(new).type(torch.FloatTensor)

        print(old.shape)
        print(rwd.shape)

        print(self.net(old))

        exit()

        """

        # We get the network output
        # out with n 'obs'
        # [act] to check ?
        # quality(?) of the action
        # matching <a1...a32> with the out
            out = torch.gather(self.net(obs_old_s.type(torch.FloatTensor)), 1, acts).squeeze(1)
            print("self.net : out --> ", out)
            exit()

            # We compute the target
            with torch.no_grad():
                # rwd = <r1...r2>
                exp = rwds + CFG.gamma * self.net(torch.FloatTensor(np.ndarray(obs_new_s))).max()

            # Compute the loss
            loss = torch.square(exp - out)
            self.loss_list.append(loss.tolist())

            # Perform a backward propagation.
            # exit() before backward prop to check
            self.opt.zero_grad()
            loss.sum().backward()
            self.opt.step()

            # stats_rwd(rwd):
        """


    def move(self, new_obs, board):
        """
        Run an epsilon-greedy policy for next action selection.
        """
        # Return random action with probability epsilon
        if random.uniform(0, 1) < CFG.epsilon or\
            len(self.obs) <= CFG.batch_size:
           return random.choice(np.flatnonzero(new_obs["action_mask"]))

        # Else, return action with highest value
        with torch.no_grad():
            # to clean up & squeeze ?..
            val = self.net(torch.tensor(new_obs['observation']).type(torch.FloatTensor))
            valid_actions = torch.tensor(np.squeeze(np.argwhere(new_obs['action_mask'] == 1).T, axis=0))
            valid_q = torch.index_select(val, 0, valid_actions)
            best_q = torch.argmax(valid_q).numpy()
            action = valid_actions[best_q].numpy()
        print("DeepK Engine Move: ", action.tolist())
        return action.tolist()



class RandomA(Agent):
    def __init__(self):
        super().__init__()

    def move(self, observation, board):
        if CFG.debug:
            print("Random Move")
        return random.choice(np.flatnonzero(observation['action_mask']))


class StockFish(Agent):
    def __init__(self):
        super().__init__()
        if CFG.engine == None:
            CFG.engine = Engine()
        self.time_to_play = CFG.time_to_play

    @staticmethod
    def move_to_act(move):
        """
        Convert a string-like move to an action.
        """
        x, y = chess_utils.square_to_coord(move.from_square)
        panel = chess_utils.get_move_plane(move)
        return (x * 8 + y) * 73 + panel

    def move(self, observation, board):
        if board.turn == False:
            board = board.mirror()
        SF_move = CFG.engine.engine.play(board=board,
                                   limit=chess.engine.Limit(time=self.time_to_play),
                                   info=chess.engine.Info(1)
                                   )
        if CFG.debug:
            print("Stockfish Engine Move: ", SF_move.move)
        return StockFish.move_to_act(SF_move.move)

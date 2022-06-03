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


class DeepKasp_Conv_Batch(Agent):
    def __init__(self):
        super().__init__()
        if CFG.agt_type == "Conv":
            self.net = network.Conv()
        if CFG.agt_type == "Linear":
            self.net = network.Linear()
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
        act = torch.stack(act).long().unsqueeze(1)
        rwd = torch.stack(rwd).type(torch.FloatTensor)
        new = torch.stack(new).type(torch.FloatTensor)

        # We get the network output
        out = torch.gather(self.net(old), 1, act).squeeze(1)
        print("self.net : out --> ", out)

        # We compute the target
        with torch.no_grad():
            exp = rwd + CFG.gamma * self.net(new).max() # TODO squeeze and get index
        print("self.net : exp --> ", exp)

        # Compute the loss
        loss = torch.square(exp - out)
        print('loss', loss, "\n")
        self.loss_list.append(loss.tolist())

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
        if random.uniform(0, 1) < CFG.epsilon or\
            len(self.obs) <= CFG.batch_size:
            if CFG.debug:
                print("Deep_K Epsilon-induced Random Move !")
            return random.choice(np.flatnonzero(new_obs["action_mask"]))

        # Else, return action with highest value # TODO FIX
        with torch.no_grad():
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
                                   limit=chess.engine.Limit(time=CFG.time_to_play), # set Depth to 1 at beginning TODO
                                   info=chess.engine.Info(1)
                                   )
        if CFG.debug:
            print("Stockfish Engine Move: ", SF_move.move)
        return StockFish.move_to_act(SF_move.move)

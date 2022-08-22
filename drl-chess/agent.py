"""
Agent module.
"""

import collections
import random
import chess.engine
import numpy as np
import torch
from pettingzoo.classic.chess import chess_utils

from config import CFG
from data import DAT
from engine import Engine
from network import Conv, DistinctLayer


class Agent:
    def __init__(self):
        pass

    def move(self, observation, board):
        pass

    def feed(self, obs_old, act, rwd, obs_new):
        pass


class DeepK(Agent):
    """
    Main agent based on a Markov Decision Process.
    Follows Deep Q-learning mechanism: Collects observation tuples from game.py and feeds samples to a NN model.
    """
    def __init__(self):
        super().__init__()

        agt_t = {"conv": Conv, "distinct": DistinctLayer}

        self.net = agt_t[CFG.net_type]()

        self.tgt = agt_t[CFG.net_type]()
        self.tgt.load_state_dict(self.net.state_dict())
        self.tgt.eval()

        self.obs = collections.deque(maxlen=CFG.buffer_size)
        self.opt = torch.optim.Adam(self.net.parameters(), lr=CFG.learning_rate)

    def feed(self, obs_old, act, rwd, obs_new):
        """
        Collect <S A R S'> observations from game and add them to the deque,
        Trigger a training each time batch is large enough.
        """
        DAT.feed_idx += 1

        old = torch.tensor(obs_old["observation"]).T
        act = torch.tensor(act)
        rwd = torch.tensor(rwd)
        new = torch.tensor(obs_new["observation"]).T

        self.obs.append((old, act, rwd, new))

        if (
            len(self.obs) >= CFG.batch_size
            and (DAT.feed_idx % (CFG.batch_size / 4) == 0)
            and CFG.train
        ):
            if CFG.debug:
                print(f"obs size: {len(self.obs)}")
            self.learn()

    def learn(self):
        """
        Train the model.
        """
        DAT.learn_idx += 1

        if CFG.debug:
            print(f"Start learn #{DAT.learn_idx}")

        batch = random.sample(self.obs, CFG.batch_size)
        old, act, rwd, new = zip(*batch)

        old = torch.stack(old).type(torch.FloatTensor)
        act = torch.stack(act).long().unsqueeze(1)
        rwd = torch.stack(rwd).type(torch.FloatTensor)
        new = torch.stack(new).type(torch.FloatTensor)

        if CFG.small_obs:
            old = old[:, 0:20, :, :]
            new = new[:, 0:20, :, :]

        # We get the network output
        out = torch.gather(self.net(old), 1, act).squeeze(1)

        # We compute the target
        with torch.no_grad():
            idx = torch.argmax(self.net(new), 1).unsqueeze(1)
            exp = rwd + CFG.gamma * torch.gather(self.tgt(new), 1, idx).squeeze(1)

        # Compute the loss
        loss = torch.square(exp - out)
        if CFG.debug:
            print("loss =", loss, "\n")
        DAT.set_loss(np.mean(loss.tolist()))  # Adds mean loss of batch.

        # Perform a backward propagation.
        self.opt.zero_grad()
        loss.sum().backward()
        ## Input Gradient Clipping.
        # torch.nn.utils.clip_grad_norm_(self.net.parameters(),
        #                               max_norm=0.3)
        self.opt.step()

        # Target Network
        if DAT.learn_idx % CFG.weight_updt == 0:
            self.tgt.load_state_dict(self.net.state_dict())
            if CFG.debug:
                print("Target network weight update")

    def move(self, new_obs, board):
        """
        Run an epsilon-greedy policy for next action selection.
        """
        # Return random action with probability epsilon
        if random.uniform(0, 1) < CFG.epsilon and CFG.train:
            if CFG.debug:
                print("Deep_K Epsilon-induced Random Move !")
            return random.choice(np.flatnonzero(new_obs["action_mask"]))

        # Else, return action with highest value
        with torch.no_grad():
            new = (
                torch.tensor(new_obs["observation"])
                .T.type(torch.FloatTensor)
                .unsqueeze(0)
            )
            if CFG.small_obs:
                new = new[:, 0:20, :, :]
            val = self.net(new).squeeze(0)
            valid_actions = torch.tensor(
                np.squeeze(np.argwhere(new_obs["action_mask"] == 1).T, axis=0)
            )
            valid_q = torch.index_select(val, 0, valid_actions)
            best_q = torch.argmax(valid_q).numpy()
            action = valid_actions[best_q].numpy()
        return action.tolist()


class Random(Agent):
    """
    Basic Agent. Always returns a random move.
    """
    def __init__(self):
        super().__init__()

    def move(self, observation, board):
        return random.choice(np.flatnonzero(observation["action_mask"]))


class StockFish(Agent):
    """
    Agent that uses the Stockfish chess engine.
    Its difficulty parameters can be set in config.py
    """
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

    def _move(self, board):
        # TODO, Maybe. Move this to the engine class to encapsulate stockfish.
        if board.turn == False:
            board = board.mirror()

        move = CFG.engine.engine.play(
            board=board,
            limit=chess.engine.Limit(time=CFG.time_to_play, depth=CFG.depth),
        )  # TODO Test Different Settings & Depths

        return self.move_to_act(move.move)

    def move(self, _, board):
        return self._move(board)


class ObservationGenerator(StockFish, DeepK):
    def __init__(self):
        super().__init__()

    def move(self, _, board):
        return super()._move(board)

    def learn(self):
        pass

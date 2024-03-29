"""
Data generator module.
"""

import numpy as np
import math

from config import CFG


class Data:
    def __init__(self):
        self.feed_idx = 0
        self.eval_idx = 0
        self.learn_idx = 0
        self.past = {0: None, 1: None}
        self.score = {0: [0], 1: [0]}

        self.stats = {
            "outcome": [],
            "loss": [0],
            "reward_1": [],
            "reward_2": [],
            "eval": [],
        }
        self.tot_win = 0
        self.tot_draw = 0

    def reset(self, hard=False):
        self.past = {0: None, 1: None}
        self.score = {0: [0], 1: [0]}
        if hard:
            self.__init__()

    def set_data(self, data, idx):
        self.past[idx] = data

    def get_data(self, idx):
        return self.past[idx]

    def set_loss(self, loss):
        """
        Save agent learning loss.
        """
        self.stats["loss"].append(loss)

    def linearize_reward(self, score):
        if score == 0:
            score = 0
        elif score < 0:
            score = math.log(-score) / math.log(10000)
            score = -score
        else:
            score = math.log(score) / math.log(10000)
        return score

    def get_reward(self, board, idx):
        old = self.score[idx][-1]
        new = CFG.engine.rating(board, idx)
        self.score[idx].append(new)
        return self.linearize_reward(new) - self.linearize_reward(old)

    def set_game(self, board):
        rwd_1 = np.diff(list(map(self.linearize_reward, self.score[0])))
        self.stats["reward_1"].append(sum(rwd_1) / len(self.score[0]))
        rwd_2 = np.diff(list(map(self.linearize_reward, self.score[1])))
        self.stats["reward_2"].append(sum(rwd_2) / len(self.score[1]))

        self.stats["outcome"].append(board.result(claim_draw=True))
        self.reset()


DAT = Data()

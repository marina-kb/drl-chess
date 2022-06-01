"""
Agent module.
"""

import random
import numpy as np
import chess.engine
from pettingzoo.classic.chess import chess_utils

from config import CFG
from engine import Engine

class Agent():
    def __init__(self):
        pass

    def move(self, observation, board):
        pass

    def feed(self, sars):
        pass


class DeepKasp(Agent):
    def __init__(self):
        super().__init__()

    def eat(self):
        pass

    def move(self, observation, board):
        pass


class RandomA(Agent):
    def __init__(self):
        super().__init__()

    def move(self, observation, board):
        print("Random Move")
        return random.choice(np.flatnonzero(observation['action_mask']))


class StockFish(Agent):
    def __init__(self):
        super().__init__()
        self.engine = Engine().engine
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
        SF_move = self.engine.play(board=board,
                                   limit=chess.engine.Limit(time=self.time_to_play),
                                   info=chess.engine.Info(1)
                                   )
        print("Stockfish Engine Move: ", SF_move.move)
        return StockFish.move_to_act(SF_move.move)

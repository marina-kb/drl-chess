"""
Agent module.
"""

import random
import os
import numpy as np
import chess.engine
from pettingzoo.classic.chess import chess_utils
# import asyncio

from config import Config


class Agent():
    def __init__(self):
        pass


class RandomA(Agent):
    def __init__(self):
        super().__init__(self)

    def move(self, observation, board):
        print("Random Move")
        return random.choice(np.flatnonzero(observation['action_mask']))




class StockFish(Agent):

    def __init__(self):
        super().__init__(self)

        SF_dir = os.path.join(os.path.dirname(__file__), '../.direnv/stockfish') # Switch to the Engine Class for connexion
        # self.transport, self.engine = await chess.engine.popen_uci(SF_dir) # need async to work
        self.engine = chess.engine.SimpleEngine.popen_uci(SF_dir)
        print("StockFish init \n")


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
                                   limit=chess.engine.Limit(time=Config().time_to_play),
                                   info=chess.engine.Info(1)
                                   )
        print("Stockfish Engine Move: ", SF_move.move)
        return StockFish.move_to_act(SF_move.move)

"""
Agent module.
"""

import random
import numpy as np
import chess.engine
import os

class Agent():

    def __init__(self):
        pass

    def move(self, observation):
        return random.choice(np.flatnonzero(observation['action_mask']))



class Engine(Agent):

    def __init__(self):
        # super().__init__(self)

        # Method for StockFish in .direnv:
        cwd = os.getcwd()
        # print(cwd)
        # self.transport, self.engine = chess.engine.popen_uci(f"{cwd}/../.direnv/stockfish")
        self.engine = chess.engine.SimpleEngine.popen_uci(f"{cwd}/../.direnv/stockfish")
        # CHANGE DIRECTORY PATH (use __file__ ?)

    def engine_move(self, board):
        return self.engine.play(board, chess.engine.Limit(time=0.1)) #Hard Code Time with config.py

    # Method to convert "E4" to Action Mask

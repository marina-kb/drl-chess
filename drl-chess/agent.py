"""
Agent module.
"""

import random
import numpy as np
import chess.engine

class Agent():

    def __init__(self):
        pass

    def move(self, observation):
        return random.choice(np.flatnonzero(observation['action_mask']))



class Engine(Agent):

    def __init__(self):
        # super().__init__(self)
        # TODO Get current directory
        self.transport, self.engine = chess.engine.popen_uci("dir + ../stockfish")

    def engine_move(self, board):
        return self.engine.play(board, chess.engine.Limit(time=0.1))

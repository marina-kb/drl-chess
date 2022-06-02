"""
Engine class to use Stockfish
"""

import os
import chess.engine

class Engine():

    def __init__(self):
        SF_dir = os.path.join(os.path.dirname(__file__), '../.direnv/stockfish')
        self.engine = chess.engine.SimpleEngine.popen_uci(SF_dir)

    def reward_calc(self, board, idx):
        """
        Use Stockfish to calculate model's reward
        """
        info = self.engine.analyse(board,
                                   limit=chess.engine.Limit(time=0.1))
        if idx == 0:
            score = info["score"].white().score(mate_score=10000)
        else:
            score = info["score"].black().score(mate_score=10000)
        return score

    def stop_engine(self):
        self.engine.quit()
        print("Stockfish stop \n")

    def calculate_model_elo(self):
        pass



# self.transport, self.engine = await chess.engine.popen_uci(SF_dir) # need async to work

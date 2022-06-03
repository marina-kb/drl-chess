"""
Engine class to use Stockfish
"""

import os
import math
import subprocess
import chess.engine

from data import DAT


class Engine:
    def __init__(self):
        SF_dir = (
            subprocess.run(["which", "stockfish"], stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip("\n")
        )
        # SF_dir = os.path.join(os.path.dirname(__file__), '../.direnv/stockfish') TODO REMOVE SF from .direnv
        self.engine = chess.engine.SimpleEngine.popen_uci(SF_dir)

        # print(self.engine.options)
        # print()
        # print(self.engine.options["Hash"])
        # print(self.engine.options["Threads"])

        # self.engine.configure({"Hash": 1024, "Threads": 8})
        self.limit = chess.engine.Limit(
            time=0.1
        )  # TODO Test 0.02 and others (and check default depth?)

    def rating(self, board, idx):
        """
        Get board rating of a player
        """
        info = self.engine.analyse(board, limit=self.limit)
        if idx == 0:
            score = info["score"].white().score(mate_score=10000)
        else:
            score = info["score"].black().score(mate_score=10000)
        return score

    def linearize(self, score):
        if score == 0:
            score = 0
        elif score < 0:
            score = math.log(-score)/math.log(10000)
            score = -score
        else:
            score = math.log(score)/math.log(10000)
        return score

    def reward(self, board, idx):
        """
        """
        old = DAT.get_score(idx)
        new = self.rating(board, idx)
        DAT.set_score(new, idx)
        return self.linearize(new) - self.linearize(old)

    def stop_engine(self):
        self.engine.quit()
        print("Stockfish stop \n")

    def calculate_model_elo(self):  # TODO PAUL A
        pass


# self.transport, self.engine = await chess.engine.popen_uci(SF_dir) # need async to work

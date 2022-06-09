"""
Engine class to use Stockfish
"""

import subprocess
import chess.engine
import os

class Engine:
    def __init__(self):
        SF_dir = (
            subprocess.run(["which", "stockfish"], stdout=subprocess.PIPE)
            .stdout.decode("utf-8").strip("\n")
            )
        # SF_dir = os.path.join(os.path.dirname(__file__), '../.direnv/stockfish')
        self.engine = chess.engine.SimpleEngine.popen_uci(SF_dir)

        # print(self.engine.options)
        # print()
        # print(self.engine.options["Hash"])
        # print(self.engine.options["Threads"])

        # self.engine.configure({"Hash": 1024, "Threads": 8})
        self.limit = chess.engine.Limit(time=0.1)
        self.engine.configure({"Skill Level": 1})
        # TODO Test 0.02 and others

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

    def stop_engine(self):
        self.engine.quit()
        print("Stockfish stop \n")

    def calculate_model_elo(self):  # TODO if possible
        pass

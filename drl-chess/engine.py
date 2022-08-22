"""
Engine class to use Stockfish
"""

import subprocess
import chess.engine


class Engine:
    def __init__(self):
        SF_dir = (
            subprocess.run(["which", "stockfish"], stdout=subprocess.PIPE)
            .stdout.decode("utf-8")
            .strip("\n")
        )
        ## If subprocess cannot find stockfish, move it to .direnv and switch to method :
        # import os
        # SF_dir = os.path.join(os.path.dirname(__file__), '../.direnv/stockfish')

        self.engine = chess.engine.SimpleEngine.popen_uci(SF_dir)

        self.limit = chess.engine.Limit(time=0.1)

        ## TODO config stockfish options to optimize performance:
        # print(self.engine.options["Hash"], self.engine.options["Threads"])
        # self.engine.configure({"Skill Level": 1,
        #                        "Threads": 8,
        #                        "Hash": 1024})

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

    def calculate_model_elo(self):  # TODO
        pass

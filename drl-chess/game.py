"""
Game Module.
"""

from pettingzoo.classic import chess_v5
import matplotlib.pyplot as plt

from data import DAT
from config import CFG


class Game:
    def __init__(self, agt):
        self.game_env = chess_v5.env()
        self.chess_env = self.game_env.env.env.env.env
        self.game_env.reset()
        self.agt = agt

    def board(self):
        return self.chess_env.board

    def play(self):
        idx = 0
        for _ in self.game_env.agent_iter(max_iter=50000):

            agt = self.agt[idx]
            dat = DAT.get_data(idx)
            new, rwd, done, info = self.game_env.last()

            if dat is not None:
                old, act = dat
                rwd = DAT.get_reward(self.board(), idx) if CFG.reward_SF else rwd
                agt.feed(old, act, rwd, new)

            if done:
                DAT.set_game(self.board())
                DAT.reset()
                break

            act = agt.move(new, self.board())
            if CFG.debug:
                print(f"Move : {act}")
            self.game_env.step(act)

            DAT.set_data((new, act), idx)

            if CFG.debug:
                print(self.board())
                print()

            idx = 1 - idx

        self.game_env.reset()


# TODO Look into using return_info at reset (from Gym lib):
#     if done:
#         observation, info = env.reset(return_info=True)
# env.close()

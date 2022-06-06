"""
Game Module.
"""

from pettingzoo.classic import chess_v5

from data import DAT, Data
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
                if CFG.reward_SF:
                    rwd = CFG.engine.reward(self.board(), idx)
                agt.feed(old, act, rwd, new)

            # if self.board().is_game_over():  ## TODO TEST DEBUG PAUL
            #     print(self.chess_env.board.outcome())
            #     Data().set_game(board=self.board())

            if done:
                DAT.set_game(self.board()) # TODO UNCOMMENT AFTER TEST
                break

            act = agt.move(new, self.board())
            if CFG.debug:
                print(f'Move : {act}')
            self.game_env.step(act)

            DAT.set_data((new, act), idx)

            if CFG.debug:
                print(self.board())
                print()

            idx = 1 - idx

        self.game_env.reset()

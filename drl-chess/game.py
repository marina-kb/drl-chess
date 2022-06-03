"""
Game Module.
"""

from pettingzoo.classic import chess_v5

from data import DAT
from config import CFG


class Game:
    def __init__(self, players):
        self.game_env = chess_v5.env()
        self.chess_env = self.game_env.env.env.env.env
        self.game_env.reset()
        self.players = players

    def board(self):
        return self.chess_env.board

    def play(self):

        history = [None, None]
        old_score = [0, 0]
        idx = 0  # Player index (0 or 1)
        for _ in self.game_env.agent_iter(max_iter=50000):

            new_obs, rwd, done, info = self.game_env.last()

            if history[idx] is not None:

                old_obs, act = history[idx]

                if CFG.reward_SF:
                    new_score = CFG.engine.reward_calc(self.board(), idx)
                    if CFG.debug:
                        print(f"{idx} relative score: {new_score}")
                    rwd = new_score - old_score[idx]
                    old_score[idx] = new_score

                self.players[idx].feed(old_obs, act, rwd, new_obs)

            if done:
                DAT.add_game(self.board().outcome(claim_draw=True).result())
                break

            move = self.players[idx].move(new_obs, self.board())
            self.game_env.step(move)

            history[idx] = new_obs, move

            if CFG.debug:
                print(self.board())
                print()

            idx = 1 - idx

        self.game_env.reset()
        return self.players[1 - idx]

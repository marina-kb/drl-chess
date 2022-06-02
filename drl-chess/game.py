"""
Game Module.
"""

from pettingzoo.classic import chess_v5

from config import CFG


class Game():

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
        idx = 0         # Player index (0 or 1)
        coup = 0
        while not self.board().is_game_over():
        # for step in self.game_env.agent_iter(max_iter=5):

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

            move = self.players[idx].move(new_obs, self.board())
            self.game_env.step(move)

            history[idx] = new_obs, move

            if CFG.debug:
                print(self.board())
                print()
            idx = 1 - idx

            coup +=1

        print(f"game over in {coup} plays!  -  winner: {self.players[1-idx]}")
        # print(self.players[idx].loss_list)
        # Add a self.game_env.reset() once game finish

        # TEMP TO GET SEC. PER COUP
        self.game_env.reset()
        return coup

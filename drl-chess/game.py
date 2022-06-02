"""
Game Module.
"""

from pettingzoo.classic import chess_v5

import numpy as np

from agent import Agent, RandomA, StockFish, DeepKasp_Conv, DeepKasp_Lin

from config import CFG

from engine import Engine



class Game():
    # Add attributes depending on type of players

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

        while not self.board().is_game_over():
        # for step in self.game_env.agent_iter(max_iter=5):

            new_obs, rwd, done, info = self.game_env.last()

            if history[idx] is not None:

                old_obs, act = history[idx]

                if CFG.reward_SF:
                    new_score = CFG.engine.reward_calc(self.board())
                    print(f"relative score: {new_score}")
                    temp_rwd = new_score - old_score[idx]
                    print(f"new - old: {temp_rwd}")
                    old_score[idx] = new_score

                self.players[idx].feed(old_obs, act, rwd, new_obs)

            move = self.players[idx].move(new_obs, self.board())
            self.game_env.step(move)

            history[idx] = new_obs, move

            print(self.board())
            print()
            idx = 1 - idx


        print(f"game over!!! winner: {self.players[1-idx]}")
        # print(self.players[idx].loss_list)
        # Add a self.game_env.reset() once game finish


# CFG.init("", rnd_seed=22)

# # players = tuple of 2 players: random / stockfish / deepk / human
# players = (DeepKasp_Conv(), StockFish())
# game = Game(players)

# game.play()

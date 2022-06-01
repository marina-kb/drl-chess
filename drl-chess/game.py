"""
Game Module.
"""

from pettingzoo.classic import chess_v5

import numpy as np

from agent import Agent, RandomA, StockFish

from config import CFG


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

        idx = 0         # Player index (0 or 1)

        while not game.board().is_game_over():
        # for step in self.game_env.agent_iter(max_iter=5):     # add step in in history to debug

            new_obs, rwd, done, info = self.game_env.last()

            if history[idx] is not None:
                old_obs, act = history[idx]
                # self.players[idx].feed(old_obs, act, rwd, new_obs)

            move = self.players[idx].move(new_obs, game.board())
            self.game_env.step(move)

            history[idx] = new_obs, move

            print(game.board())
            print()
            idx = 1 - idx


        print("game over!!!")
        # Add a self.game_env.reset() once game finish



# players = tuple of 2 players: random / stockfish / deepk / human
players = (RandomA(), StockFish())
game = Game(players)

game.play()

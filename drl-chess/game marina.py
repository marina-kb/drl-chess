"""
Game Module.
"""

import re
from typing_extensions import Self
from pettingzoo.classic import chess_v5
import random
import numpy as np
from pettingzoo.classic.chess import chess_utils

from agent import Agent, StockFish

import hashlib

# # Encoding function
# def encode_state(observation):
#     # encode observation as bytes
#     obs_bytes = str(observation).encode('utf-8')
#     # create md5 hash
#     m = hashlib.md5(obs_bytes)
#     # return hash as hex digest
#     state = m.hexdigest()
#     return(state)

class Game():

    # Add attributes depending on type of players
    # players = tuple of 1 or 2 types of players: random / stockfish / deepk / human
    def __init__(self, players):
        self.game_env = chess_v5.env()
        self.chess_env = self.game_env.env.env.env.env
        self.game_env.reset()
        self.players = players

    def board(self):
        return self.chess_env.board

    @staticmethod
    def move_to_act(move):
        """
        Convert a string-like move to an action.
        """
        x, y = chess_utils.square_to_coord(move.from_square)
        panel = chess_utils.get_move_plane(move)
        return (x * 8 + y) * 73 + panel

# Version combined
# iters to change from Config()
    def play(self, iters):
        player_idx = 0
        self.game_env.agents = self.players
        #observations = []
        for agent in self.game_env.agent_iter(max_iter=iters):
            board = Game.board(self)
            print(agent)
            if agent == 'player_0':
                board = Game.board(self)
                print(board)
            else:
                board = board.mirror()
                print(board)

            observation, reward, done, info = self.game_env.last()

            #print(type(self.players[player_idx]))
            if str(type(self.players[player_idx])) == "<class 'agent.Agent'>":
                move = self.players[player_idx].move(observation)
                print('random plays!')


            elif str(type(self.players[player_idx])) == "<class 'agent.StockFish'>":
                board = board.mirror()
                print('SF plays!')
                print(self.players[player_idx].move(game.board()))
                move = self.players[player_idx].move(game.board())
                move = Game.move_to_act(move)
                print(move)

            else:
                # to add deep_k & human?
                pass

            player_idx = 1 - player_idx
            self.game_env.step(move)
            #print(game.board())

# Ajouter un historique des n derniers coups (observations, rewards, etc.)

# players = tuple of 1 or 2 types of players: random / stockfish / deepk / human

players = (Agent(), StockFish())
game = Game(players)

game.play(10)
print(game.board())

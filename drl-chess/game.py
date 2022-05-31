"""
Game Module.
"""

import re
from pettingzoo.classic import chess_v5
import random
import numpy as np

from agent import Agent, Engine

import hashlib

# Encoding function
def encode_state(observation):
    # encode observation as bytes
    obs_bytes = str(observation).encode('utf-8')
    # create md5 hash
    m = hashlib.md5(obs_bytes)
    # return hash as hex digest
    state = m.hexdigest()
    return(state)


class Game():

    def __init__(self):
        self.game_env = chess_v5.env()
        self.chess_env = self.game_env.env.env.env.env
        self.game_env.reset()

    def board(self):
        return self.chess_env.board

    # Version STOCKFISH/Random 1/2
    def play(self, iters):
        agent = Agent()
        for agent in self.game_env.agent_iter(max_iter=iters):
            if agent == 'player_0':
                observation, reward, done, info = self.game_env.last()
                move = agent.move(observation)
                self.game_env.step(move)
            else:
                observation, reward, done, info = self.game_env.last()
                move = Engine().move(game.board())
                self.game_env.step(move)


    # Version Basic Random
    def play(self):
        for agent in self.game_env.agent_iter(max_iter=300):
            observation, reward, done, info = self.game_env.last()
            move = Agent().move(observation)
            self.game_env.step(move)

        ## To encode endgame board state:
            # state = encode_state(self.game_env.render(mode = 'ansi'))
            # print(state)













game = Game()

game.play(3)
print(game.board())

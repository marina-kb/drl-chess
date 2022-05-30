"""
Game Module.
"""

from pettingzoo.classic import chess_v5

class Game():

    def __init__(self):
        self.game_env = chess_v5.env()
        self.chess_env = self.game_env.env.env.env.env
        self.game_env.reset()

    def board(self):
        return self.chess_env.board

    def play(self):
        for agent in self.game_env.agent_iter():
            observation, reward, done, info = self.game_env.last()
            move = ?
            self.game_env.step(move)


game = Game()
game.play()

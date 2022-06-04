"""
Data generator module.
"""


class Data:
    def __init__(self):
        """
        Declare types but do not instanciate anything
        """
        self.loss = []
        self.game = []
        self.past = {0: None, 1: None}
        self.score = {0: [0], 1: [0]}

    def set_data(self, data, idx):
        """
        Save past observation and action of agent.
        """
        self.past[idx] = data

    def get_data(self, idx):
        """
        Get past observation and action of agent.
        """
        return self.past[idx]

    def set_loss(self, loss):
        """
        Save agent learning loss.
        """
        self.loss.append(loss)

    def set_game(self, board):
        """
        Save the current game outcome.
        """
        self.game.append(board.outcome(claim_draw=True).result())  # TODO DEBUG PAUL BROKEN?


    def set_score(self, score, idx):
        self.score[idx].append(score)

    def get_score(self, idx):
        return self.score[idx][-1]

    def reset(self):
        self.__init__()


DAT = Data()

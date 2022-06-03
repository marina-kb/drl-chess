class Data:
    def __init__(self):
        """
        Declare types but do not instanciate anything
        """
        self.loss = []
        self.game = []

    def add_loss(self, loss):
        self.loss.append(loss)

    def add_game(self, game):
        self.game.append(game)

    def reset(self):
        self.__init__()


DAT = Data()

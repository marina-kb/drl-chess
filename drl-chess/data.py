class Data:
    def __init__(self):
        """
        Declare types but do not instanciate anything
        """
        self.loss = []

    def add_loss(self, loss):
        self.loss.append(loss)

    def reset(self):
        self.__init__()


DAT = Data()

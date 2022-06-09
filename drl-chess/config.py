"""
Configuration Module.
This module defines a singleton-type configuration class that can be used all across our project. This class can contain any parameter that one may want to change from one simulation run to the other.
"""

import random
import engine


class Configuration:
    """
    This configuration class is extremely flexible due to a two-step init process.
    We only instanciate a single instance of it (at the bottom if this file) so that all modules can import this singleton at load time.
    The second initialization (which happens in main.py) allows the user to input custom parameters of the config class at execution time.
    """

    def __init__(self):
        """
        Declare types but do not instanciate anything
        """
        self.alpha = 0.2
        self.gamma = 1
        self.epsilon = None
        self.epsilon_decay = 0.01

        self.learning_rate = 0.1
        self.max_norm = 0.3

        self.rnd_seed = None
        self.net_type = None

        self.reward_SF = True
        self.engine = None
        self.time_to_play = 0.1
        self.depth = 2

        self.buffer_size = 10000
        self.batch_size = 1024
        self.weight_updt = 50

        self.debug = False

        self.small_obs = False

        self.train = True


    def init(self, net_type, **kwargs):
        """
        User-defined configuration init. Mandatory to properly set all configuration parameters.
        """

        # Mandatory arguments go here. In our case it is useless.
        self.net_type = net_type

        # We set default values for arguments we have to define
        self.rnd_seed = 1   # for diff use: random.randint(0, 1000)
        self.epsilon = 0.05

        # However, these arguments can be overriden by passing them as keyword arguments in the init method. Hence, passing for instance epsilon=0.1 as a kwarg to the init method will override the default value we just defined.
        self.__dict__.update(kwargs)

        # Once all values are properly set, use them.
        random.seed(self.rnd_seed)

        # import engine

        # Adding Stockfish engine:
        if self.reward_SF:
            self.engine = engine.Engine()
            if CFG.debug:
                print("Stockfish init \n")


CFG = Configuration()

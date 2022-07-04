"""
Configuration Module.
This module defines a singleton-type configuration class that can be used all across our project.
This class can contain any parameter that one may want to change from one simulation run to the other.
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
        self.net_type = None       # Network variable

        self.train = True          # Allow network training

        self.learning_rate = 0.1   # Network Learning Rate
        self.gamma = 1             # Reward discount factor
        self.epsilon = None        # Epsilon variable
        self.epsilon_decay = 0.01  # Epsilon decay rate
        self.weight_updt = 50      # Target network activation frequency

        self.small_obs = False     # Use size-20 obs
        self.buffer_size = 10000   # Obs deque size
        self.batch_size = 1024     # Obs batch size

        self.reward_SF = True      # Use move-by-move rewards
        self.engine = None         # Stockfish variable
        self.time_to_play = 0.1    # Stockfish config
        self.depth = 2             # Stockfish config

        self.rnd_seed = None       # Random seed

        self.debug = False         # Print board and infos


    def init(self, net_type, **kwargs):
        """
        User-defined configuration init. Mandatory to properly set all configuration parameters.
        """

        # Mandatory arguments go here. In our case it is useless.
        self.net_type = net_type

        # We set default values for arguments we have to define
        self.rnd_seed = 1  # for diff use: random.randint(0, 1000)
        self.epsilon = 0.05

        # However, these arguments can be overriden by passing them as keyword arguments in the init method.
        # Hence, passing for instance epsilon=0.1 as a kwarg to the init method will override the default value we just defined.
        self.__dict__.update(kwargs)

        # Once all values are properly set, use them.
        random.seed(self.rnd_seed)

        # Adding Stockfish engine:
        if self.reward_SF:
            self.engine = engine.Engine()
            if CFG.debug:
                print("Stockfish init \n")


CFG = Configuration()

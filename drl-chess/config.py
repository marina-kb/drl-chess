"""
Config class for game settings
"""

class Config():

    def __init__(self, time_to_play=0.1, max_iters=10, opp_start_black=True):
        self.time_to_play = time_to_play
        self.max_iters = max_iters
        self.opp_start_black = opp_start_black
        # Add new config params

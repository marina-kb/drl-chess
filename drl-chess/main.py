from time import time
import game
import agent
from config import CFG
import math


def main():


    CFG.init(
        net_type="Conv",
        reward_SF=False,
    )

    players = (agent.StockFish(), agent.Random())
    environment = game.Game(players)

    for n in range(5):
        CFG.epsilon = math.exp(-CFG.epsilon_decay*n)
        environment.play()

    if CFG.engine is not None:
        CFG.engine.stop_engine()


def gen_data():

    CFG.init(net_type="conv", debug=False, reward_SF=True)

    agt = (agent.ObservationGenerator(), agent.ObservationGenerator())
    env = game.Game(agt)
    obs = []

    for _ in range(10):
        env.play()

        obs += agt[0].obs
        obs += agt[1].obs

        agt[0].obs.clear()
        agt[1].obs.clear()

# TODO create utils.py and then a function to_disk that pickles dataset.

gen_data()

# def save_obs()

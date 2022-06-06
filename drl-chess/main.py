import math
import os

import game
import agent
from config import CFG
from data import DAT
import utils


def stop_eng():
    if CFG.engine is not None:
        CFG.engine.stop_engine()


def main(agt=None):

    CFG.init(net_type="conv", reward_SF=True, debug=True)

    if agt is None:
        agt = (agent.DeepK(), agent.Random())
    env = game.Game(agt)

    for n in range(1):
        # CFG.epsilon = math.exp(-CFG.epsilon_decay * n)
        env.play()

    stop_eng()


def gen_data():

    CFG.init(net_type="conv", debug=False, reward_SF=True)

    agt = (agent.ObservationGenerator(), agent.ObservationGenerator())
    env = game.Game(agt)
    obs = []

    for _ in range(25):
        env.play()

        obs += agt[0].obs
        obs += agt[1].obs

        agt[0].obs.clear()
        agt[1].obs.clear()

    utils.to_disk(obs) # Push obs batch to ../data/


def load_agent():

    CFG.init(net_type="conv", debug=False, reward_SF=True)

    agt = agent.DeepK()
    dir = os.path.join(os.path.dirname(__file__), f'../data-test')

    for dir in utils.get_files(dir):
        print(dir)
        for obs in utils.from_disk(dir):
            agt.obs.append(obs)
            if len(agt.obs) == 32:
                print("train")
                agt.learn()
                agt.obs = []
    return agt


while True:
    gen_data()

# for _ in range(10):
#     gen_data()
# stop_eng()

# main()

# load_agent()

from time import time
import game
import agent
from config import CFG
from data import DAT
import math
from utils import to_disk


def quit():
    if CFG.engine is not None:
        CFG.engine.stop_engine()

def main():

    CFG.init(net_type="conv", reward_SF=True)

    agt = (agent.DeepK(), agent.Random())
    env = game.Game(agt)

    for n in range(1):
        # CFG.epsilon = math.exp(-CFG.epsilon_decay * n)
        env.play()

def gen_data():

    CFG.init(net_type="conv", debug=False, reward_SF=True)

    agt = (agent.ObservationGenerator(), agent.ObservationGenerator())
    env = game.Game(agt)
    obs = []

    for _ in range(100):
        env.play()

        obs += agt[0].obs
        obs += agt[1].obs

        agt[0].obs.clear()
        agt[1].obs.clear()
        # print('game done')

    # PUSH TO DATA/
    to_disk(obs)
    # print('saving')


# for _ in range(1):
#     gen_data()


while True:
    gen_data()

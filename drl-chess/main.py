import math
import game
import agent
from config import CFG
from data import DAT
from utils import to_disk


def stop_eng():
    if CFG.engine is not None:
        CFG.engine.stop_engine()


def main():

    CFG.init(net_type="conv", reward_SF=True, debug=False)

    agt = (agent.DeepK(), agent.StockFish())
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

    to_disk(obs) # Push obs batch to ../data/


# while True:
#     gen_data()

for _ in range(10):
    gen_data()
    stop_eng()

# main()

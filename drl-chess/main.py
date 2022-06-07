import math
import os
import matplotlib.pyplot as plt

import game
import agent
from config import CFG
from data import DAT
import utils



def stop_eng():
    if CFG.engine is not None:
        CFG.engine.stop_engine()


def main(agt=None):

    CFG.init(net_type="conv", reward_SF=True, debug=False)

    if agt is None:
        agt = (agent.DeepK(), agent.StockFish())
    env = game.Game(agt)

    for n in range(100):
        CFG.epsilon = math.exp(-CFG.epsilon_decay * n)
        print(f"Playing game {n}")
        env.play()
        # print(f"outcome : {DAT.stats['outcome'][-1]}")
        print(f"loss : {DAT.stats['loss'][-1]} \n")

    stop_eng()


def gen_data():

    CFG.init(net_type="conv", debug=False, reward_SF=True)

    agt = (agent.ObservationGenerator(), agent.ObservationGenerator())
    env = game.Game(agt)
    obs = []

    for _ in range(50):
        env.play()

        obs += agt[0].obs
        obs += agt[1].obs

        agt[0].obs.clear()
        agt[1].obs.clear()

    utils.to_disk(obs) # Push obs batch to ../data/


def load_agent():

    CFG.init(net_type="conv", debug=False, reward_SF=True)

    agt = agent.DeepK()
    dir = os.path.join(os.path.dirname(__file__), f'../data')

    print(f" files to load: {len(utils.get_files(dir))}")

    for dir in utils.get_files(dir):
        print(dir)
        for obs in utils.from_disk(dir):
            agt.obs.append(obs)
            if len(agt.obs) % 32 == 0:
                agt.learn()
                agt.obs = []
        print(f"loss : {sum(DAT.stats['loss']) / len(DAT.stats['loss'])}")
    return agt


# while True:
#     gen_data()

# for _ in range(10):
#     gen_data()
# stop_eng()

load_agent()

# main()

# stats = True

if stats:
    fig = plt.figure(figsize=(15, 10))  # TODO Move to utils

    # Loss subplot
    plt.subplot(2,2,1)
    plt.plot(DAT.stats['loss'], label='mean loss')
    plt.title('mean loss')
    plt.legend()

    # Reward player 0 subplot
    plt.subplot(2,2,3)
    plt.plot(DAT.stats['reward_1'], label='mean_rwd_DeepK', c='black')
    # plt.ylim(-1,1)
    plt.title("mean reward DeepK")

    # Reward player 1 subplot
    plt.subplot(2,2,4)
    plt.plot(DAT.stats['reward_2'], label='mean_rwd_SF', c='black')
    # plt.ylim(-1,1)
    plt.title("mean reward SF")

    # Global figure methods
    plt.suptitle('loss&rwd')
    plt.show()

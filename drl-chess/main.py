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

    for n in range(200):
        CFG.epsilon = math.exp(-CFG.epsilon_decay * n)
        print(f"Playing game {n}")
        env.play()

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
    dir = os.path.join(os.path.dirname(__file__), f'../data-test')

    for dir in utils.get_files(dir):
        print(dir)
        for obs in utils.from_disk(dir):
            agt.obs.append(obs)
            if len(agt.obs) == CFG.batch_size:
                print("train")
                agt.learn()
                agt.obs = []
    return agt


# while True:
#     gen_data()

# for _ in range(10):
#     gen_data()
# stop_eng()

main()

fig = plt.figure(figsize=(15, 10))
plt.subplot(2,2,1)
plt.plot(DAT.stats['loss'], label='mean loss')
plt.title('mean loss')
plt.legend()
# Second subplot
plt.subplot(2,2,3)
plt.plot(DAT.stats['reward_1'], label='mean_rwd_DeepK', c='black')
# plt.ylim(-1,1)
plt.title("cumulative reward DeepK")
# Second subplot
plt.subplot(2,2,4)
plt.plot(DAT.stats['reward_2'], label='mean_rwd_SF', c='black')
# plt.ylim(-1,1)
plt.title("cumulative reward SF")
# Global figure methods
plt.suptitle('loss&rwd')
plt.show()

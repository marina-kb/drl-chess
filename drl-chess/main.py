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

    CFG.init(net_type="conv", reward_SF=True, debug=False, small_obs=False)

    if agt is None:
        agt = (agent.DeepK(), agent.StockFish())
    env = game.Game(agt)

    for n in range(5):
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

    CFG.init(net_type="conv", debug=False, reward_SF=True, depth=1)

    agt = agent.DeepK()
    dir = os.path.join(os.path.dirname(__file__), f'../data')
    for dir in utils.get_files(dir):
        print(dir)
        for obs in utils.from_disk(dir):
            agt.obs.append(obs)
            print(len(agt.obs))
            if len(agt.obs) == CFG.batch_size:
                print("train")
                agt.learn()
                agt.obs = []
                if DAT.learn_idx % 10 == 0:
                    eval(agt)

    return agt


def eval(agt, n_eval=3):
    eps, CFG.epsilon = CFG.epsilon, 0
    agt.net.eval()
    env = game.Game((agt, agent.StockFish()))
    for _ in range(n_eval):
        env.play()
        print('playing')
    winner = DAT.stats['outcome'][:-n_eval]
    wins = winner.count('1-0')
    draw = winner.count('1/2-1/2')
    print(f'{agt} a gagné {wins}/{n_eval} et a fait {draw} nuls')
    agt.net.train()
    CFG.epsilon = eps




# while True:
#     gen_data()

# for _ in range(10):
#     gen_data()
# stop_eng()

load_agent()

# main()

stats = False

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

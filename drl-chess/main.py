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

    CFG.init(net_type="conv", debug=False, reward_SF=False, depth=1, small_obs=True) # Check reward SF dependencies

    agt = agent.DeepK()
    dir = os.path.join(os.path.dirname(__file__), f'../data')
    for dir in utils.get_files(dir):
        print(dir)
        for obs in utils.from_disk(dir):
            agt.obs.append(obs)
            if len(agt.obs) >= CFG.batch_size:
                agt.learn()
                if DAT.learn_idx % 50 == 0:
                    print(f"Training loss: {DAT.stats['loss'][-1]}")
                    eval(agt)
                agt.obs = []

    return agt


def eval(agt, n_eval=5):
    CFG.train = False
    agt.net.eval()
    env = game.Game((agt, agent.StockFish()))

    for _ in range(n_eval):
        env.play()
    winner = DAT.stats['outcome'][-n_eval:]
    wins = winner.count('1-0')
    draw = winner.count('1/2-1/2')
    print(f'Wins {wins}, Losses {5 - wins - draw}, Draws {draw} \n')
    if wins > n_eval/2:
        print("KASPAROV")

    agt.net.train()
    CFG.train=True




# while True:
#     gen_data()

# for _ in range(10):
#     gen_data()
# stop_eng()

# load_agent()

main()

utils.plot_stats(DAT)

import math
import os
import matplotlib.pyplot as plt

from pettingzoo.classic.chess.chess_utils import result_to_int

import game
import agent
from config import CFG
from data import DAT
import utils


def stop_eng():
    if CFG.engine is not None:
        CFG.engine.stop_engine()


def main(agt=None):

    CFG.init(net_type='conv', debug=False, reward_SF=False,  small_obs=False,
             learning_rate = 0.1, depth=4, time_to_play=0.1)

    if agt is None:
        agt = (agent.DeepK(), agent.StockFish())         #original agt
        #agt = (utils.w8_loader(agent.DeepK.net,'pickled_model'), agent.StockFish()) # weight loader
    env = game.Game(agt)

    for n in range(100000):
        CFG.epsilon = math.exp(-CFG.epsilon_decay * n)
        env.play()

        if n % 150 == 0 and n != 0:
            print(f"loss : {DAT.stats['loss'][-1]}")
            eval(agt[0])


def gen_data():

    DAT.reset(True)
    CFG.init(net_type="conv", debug=False, reward_SF=False)

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

    CFG.init(net_type="conv", debug=False, reward_SF=False, small_obs=False,   # Check reward SF dependencies
             learning_rate = 0.01)

    agt = agent.DeepK()
    dir = os.path.join(os.path.dirname(__file__), f'../data')

    for l in range(100):
        print(f'loop #{l}')
        for idx, dir in enumerate(utils.get_files(dir)):
            print(dir)
            for obs in utils.from_disk(dir):
                agt.obs.append(obs)
                if len(agt.obs) >= CFG.batch_size:
                    agt.learn()
                    agt.obs = []
            print(f"Training loss: {DAT.stats['loss'][-1]}")
            if idx % 5 == 0 and idx != 0:
                eval(agt)

    utils.w8_saver(agt, 'pickledmodel')

    # main(agt = (agt, agent.StockFish()) )


def eval(agt, n_eval=5):
    DAT.eval_idx += 1
    CFG.train = False
    CFG.depth = 1
    agt.net.eval()
    env = game.Game((agt, agent.Random()))

    for _ in range(n_eval):
        env.play()

    win = list(map(result_to_int, DAT.stats['outcome'][-n_eval:]))
    DAT.stats['eval'].append((win.count(0), win.count(1), win.count(-1)))
    print(f'{DAT.eval_idx}: Wins {win.count(1)}, Draws {win.count(0)}, Losses {win.count(-1)} \n')
    if sum(win) > 0:
        print("KASPAROV")

    DAT.tot_win += win.count(1)
    DAT.tot_draw += win.count(0)
    print(f'Since init: total wins {DAT.tot_win} & total draws {DAT.tot_draw}')

    agt.net.train()
    CFG.train=True
    CFG.depth = 5
    agt.obs = []



# while True:
#     gen_data()


load_agent()


# main()


# utils.plot_stats(DAT)

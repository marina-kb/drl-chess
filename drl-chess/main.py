import math
import os
import sys

from pettingzoo.classic.chess import chess_utils

import agent
import game
import utils
from config import CFG
from data import DAT


def stop_eng():
    if CFG.engine is not None:
        CFG.engine.stop_engine()


def main(agt=None):

    CFG.init(
        net_type="conv",
        debug=False,
        reward_SF=False,
        small_obs=True,
        learning_rate=0.01,
        depth=1,
        time_to_play=0.1,
    )

    if agt is None:
        agt = (agent.DeepK(), agent.StockFish())  # original agt
        ## To load an agent with saved state from disk, use below instead:
        # agt = (utils.w8_loader(agent.DeepK.net,'pickled_model'), agent.StockFish())

    env = game.Game(agt)

    for n in range(1000):
        CFG.epsilon = math.exp(-CFG.epsilon_decay * n)
        env.play()

        if n % 50 == 0 and n != 0:
            print(f"loss : {DAT.stats['loss'][-1]}")
            eval(agt[0])

    stop_eng()


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

    utils.to_disk(obs)  # Push obs batch to ../data/
    stop_eng()


def load_agent():

    CFG.init(
        net_type="conv",
        debug=False,
        reward_SF=False,
        small_obs=False,  # TODO Check reward SF dependencies
        learning_rate=0.01,
    )

    agt = agent.DeepK()
    dir = os.path.join(os.path.dirname(__file__), "../data")

    for l in range(100):
        print(f"loop #{l}")

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

        ## Unindent below by 1 to only save once looped through all files
        utils.w8_saver(agt, f"pickledmodel-{l}")

    ## Uncomment below to keep training with main once looped through all files:
    # main(agt = (agt, agent.StockFish()))

    stop_eng()


def eval(agt, n_eval=5):

    DAT.eval_idx += 1

    CFG.train = False
    CFG.depth = 1
    agt.net.eval()  # Set NN model to evaluation mode.

    env = game.Game((agt, agent.StockFish()))

    for _ in range(n_eval):
        env.play()

    win = list(map(chess_utils.result_to_int, DAT.stats["outcome"][-n_eval:]))
    DAT.stats["eval"].append((win.count(0), win.count(1), win.count(-1)))
    print(
        f"{DAT.eval_idx}: Wins {win.count(1)}, Draws {win.count(0)}, Losses {win.count(-1)} \n"
    )
    if sum(win) > 0:
        print("KASPAROV")

    DAT.tot_win += win.count(1)
    DAT.tot_draw += win.count(0)
    print(f"Since init: total wins {DAT.tot_win} & total draws {DAT.tot_draw}")

    agt.net.train()  # Set NN model back to training mode
    CFG.train = True
    CFG.depth = 5


if __name__ == "__main__":
    """
    SWITCH DEPENDING ON USE
    """
    if len(sys.argv) > 1:
        if sys.argv[1] == "main":
            main()
        if sys.argv[1] == "gen":
            while True:
                gen_data()
        if sys.argv[1] == "load":
            load_agent()
    else:
        main()

    # utils.plot_hist()

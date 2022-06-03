from time import time
import game
import agent
from config import CFG


def main():

    # Init chess game congig
    CFG.init(
        net_type="Conv",
        rnd_seed=22,  # Pick a random seed
        epsilon=0.05,  # Treshold for Epsilon-Greedy policy
        debug=True,  # Print board and stuff to debug
        reward_SF=False,  # Choose Reward system ('PettingZoo' or 'Stockfish')
    )

    # Init 2 agents
    players = (agent.StockFish(), agent.RandomA())

    # Init Chess Game Environment
    environment = game.Game(players)

    coups = 0
    winner_list = []
    n_games = 5

    # TODO Add a decreasing Epsilon Policy by game

    start_time = time()  ### TIME START
    for _ in range(n_games):
        # coups += environment.play()
        winner_list.append(environment.play())

    timing = round((time() - start_time), 5)  ### TIME END

    print("\n--- Loop done in %s seconds (avg %s sec)--- " % (timing, timing / n_games))
    # print("--- Total coups: %s / Seconds per coup: %s--- \n" % (coups, (timing/n_games/coups)))

    print("WINNERS:", winner_list)

    # TODO Pipeline to save data
    # print(len(players[0].loss_list))

    if CFG.engine is not None:
        CFG.engine.stop_engine()


def gen_data():

    # Init chess game congig
    CFG.init(net_type="conv")

    players = (agent.ObservationGenerator(), agent.ObservationGenerator())
    environment = game.Game(players)

    environment.play()

    # TODO Generate stockfish obs samples
    pass


gen_data()

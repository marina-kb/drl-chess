from time import time
import game
import agent
from config import CFG

# Init chess game congig
CFG.init(agt_type='Conv',
         rnd_seed=22,           # Pick a random seed
         epsilon=0.05,          # Treshold for Epsilon-Greedy policy
         debug=False,           # Print board and stuff to debug
         reward_SF=True         # Choose Reward system ('PettingZoo' or 'Stockfish')
         )

# Init 2 agents
players = (agent.DeepKasp_Conv(),
           agent.StockFish()
           )

# Init Chess Game Environment
environment = game.Game(players)

start_time = time()
coups = 0
n_games = 1
for _ in range(n_games):
    coups += environment.play()
timing = round((time() - start_time), 5)
print("\n--- Loop done in %s seconds (avg %s sec)--- " % (timing, timing/n_games))
print("--- Total coups: %s / Seconds per coup: %s--- \n" % (coups, (timing/n_games/coups)))


# Saving data TO CODE
print(len(players[0].loss_list))





if CFG.engine is not None:
    CFG.engine.stop_engine()

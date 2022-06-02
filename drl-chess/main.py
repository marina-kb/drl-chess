from time import time
import game
import agent
from config import CFG

# Init chess game congig
CFG.init("",
         rnd_seed=22,       # Pick a random seed
         epsilon=0.05,      # Choose a
         debug=False,        # Print board and stuff to debug
         reward_SF=True     # Choose Reward system ('PettingZoo' or 'Stockfish')
         )

# Init 2 agents
players = (agent.DeepKasp_Conv(),
           agent.StockFish()
           )

# Init Chess Game Environment
start_time = time()
for _ in range(10):
    environment = game.Game(players)
    environment.play()
print("--- Loop done in %s seconds ---" % (time() - start_time))

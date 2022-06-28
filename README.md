# Learning Chess through Deep Reinforcment Learning

## About

_This is a 10-day project from Le Wagon's ML course._

We use the PettingZoo library for our agent's reinforcement learning process:
https://github.com/Farama-Foundation/PettingZoo

We also introduce the possibility to calculate move-by-move rewards using the Stockfish chess engine (instead of game-by-game rewards from PettingZoo Classic):
https://github.com/official-stockfish/Stockfish


Settup
-------
### Clone the repo and install libraries
```bash
pip install -r requirements.txt
```

### Install Stockfish
With Homebrew:
```bash
brew install stockfish
```

Otherwise:
```bash
sudo apt-get -y install stockfish
```

### Basic Usage
Run `main.py`. With the `main()` function, you can choose two agents in the `agt` key and modify various game parameters inside `CFG.init()`.

You can also generate multiple batches of games to be saved as .pkl with the `gen_data()` function. Then, train and evaluate your model with this pre-generated data using the `load_agent()` function.

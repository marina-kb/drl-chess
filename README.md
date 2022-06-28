# Learning Chess through Deep Reinforcment Learning

## About

_This is a 10-day project from Le Wagon's ML course._

We use the PettingZoo library for our agent's reinforcement learning process:
https://github.com/Farama-Foundation/PettingZoo

We also introduce the possibility to calculate move-by-move rewards using the Stockfish chess engine (instead of game-by-game rewards from PettingZoo Classic):
https://github.com/official-stockfish/Stockfish


How to use
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
sudo apt-get update && sudo apt-get -y install stockfish
```

### Basic Usage
Run `python main.py`, you can modify various game parameters inside `CFG.init()` and choose the two agents using the `agt` key.

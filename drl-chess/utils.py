"""
Save info in pickle file
"""

import pickle
from datetime import datetime
import os
import glob


def to_disk(obs):

    pdt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir = os.path.join(os.path.dirname(__file__), f'../data/{pdt}_databatch.pkl')

    with open(dir, 'wb') as file:
        pickle.dump(obs, file)

    print(f"Save to pickle @ {pdt}")


def from_disk(file):
    with open(file, 'rb') as f:
        return pickle.load(f)


def get_files(dir):
    return glob.glob(dir + '/*.pkl')

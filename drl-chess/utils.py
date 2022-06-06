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


def load_from_file(file):

    dir = os.path.join(os.path.dirname(__file__), f'../data/*.pkl')   # *_databatch.pkl

    with open(dir, 'rb') as f:
        x = pickle.load(f)

    print(x)


def load_multiple():

    dir = os.path.join(os.path.dirname(__file__), f'../data-test')
    file_list = glob.glob(dir + '/*.pkl')

    full_pkl = []

    for file_path in file_list:
        with open(file_path, 'rb') as f:
            full_pkl += [obs for obs in pickle.load(f)]
    print(len(full_pkl), len(full_pkl[0]))
    old, act, rwd, new = zip(*full_pkl)
    print(rwd)

load_multiple()

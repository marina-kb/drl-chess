"""
Save info in pickle file
"""

import pickle
from datetime import datetime
import os


def to_disk(obs):

    pdt = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    dir = os.path.join(os.path.dirname(__file__), f'../data/{pdt}_databatch.pkl')

    with open(dir, 'wb') as file:
        pickle.dump(obs, file)

    print(f"Save to pickle @ {pdt}")


def load_from_file(file):

    dir = os.path.join(os.path.dirname(__file__), f'../data/{file}')   # *_databatch.pkl

    with open(dir, 'rb') as f:
        x = pickle.load(f)
    print(x)

# load_from_file('2022-06-04_10/32/19-pretraining.pkl')

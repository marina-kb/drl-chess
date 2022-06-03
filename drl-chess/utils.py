"""
Save info in pickle file
"""

import pickle
from datetime import datetime

# from main import gen_data


def to_disk(obs):
    pdt = datetime.now().strftime('%Y-%m-%d %H:%M:%S.%f')[:-3]
    with open(f'../data/{pdt}-pretraining.pkl', 'wb') as file:
        pickle.dump(obs, file)

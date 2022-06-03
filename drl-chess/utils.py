"""
Save info in pickle file
"""

import pickle
# from main import gen_data


def to_disk(obs):
    with open('../data/pretraining.pkl', 'wb') as file:
        pickle.dump(obs, file)

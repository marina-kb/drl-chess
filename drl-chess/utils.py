"""
Save info in pickle file
"""

import pickle
from datetime import datetime
import os
import glob
import matplotlib.pyplot as plt


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


def plot_stats(DAT, figsize=(15,10)):

    fig = plt.figure(figsize=figsize)  # TODO Move to utils

    # Loss subplot
    plt.subplot(2,2,1)
    plt.plot(DAT.stats['loss'], label='mean loss')
    plt.title('mean loss')
    plt.legend()

    # Reward player 0 subplot
    plt.subplot(2,2,3)
    plt.plot(DAT.stats['reward_1'], label='mean_rwd_DeepK', c='black')
    # fig.ylim(-1,1)
    plt.title("cumulative reward DeepK")

    # Reward player 1 subplot
    plt.subplot(2,2,4)
    plt.plot(DAT.stats['reward_2'], label='mean_rwd_SF', c='black')
    # plt.ylim(-1,1)
    plt.title("cumulative reward SF")

    # Global figure methods
    plt.suptitle('loss&rwd')
    plt.show()

#Viannou va écrire une super fonction ci-dessous

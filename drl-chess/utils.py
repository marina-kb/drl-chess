"""
Save info in pickle file
"""

import pickle
from datetime import datetime
import os
import glob
import matplotlib.pyplot as plt
import torch
import pandas as pd
from data import DAT


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

    fig = plt.figure(figsize=figsize)

    # Loss subplot
    plt.subplot(2,2,1)
    plt.plot(DAT.stats['loss'], label='mean loss')
    plt.title('mean loss')
    plt.legend()

    # Win track
    plt.subplot(2,2,2)
    info = DAT.stats['outcome']
    l=[int(i[0]) for i in info]
    plt.plot(l)
    #plt.plot(DAT.stats['outcome'])
    plt.title("Win")

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

    # info = DAT.stats['outcome']
    # l=[int(i[0]) for i in info]
    # plt.subplot(2,2,4)
    # sns.histplot(data=l, x='value', hue='val_type', multiple='dodge', discrete=True,
    #       edgecolor='white', palette=plt.cm.Accent, alpha=1)



    # Global figure methods
    plt.suptitle('loss&rwd')
    plt.show()
    print(DAT.stats['outcome'])


def plot_hist():

    drw, win, loss = zip(*DAT.stats['eval'])

    idx = []
    for i in range(len(win)):
        idx.append(f"Eval Group {i+1}")

    results_df = pd.DataFrame(data={'Wins':win, 'Draws':drw, '--':loss},
                                index=idx)

    # print(results_df)

    fig, ax0 = plt.subplots(nrows=1, ncols=1)
    colors = ['green', 'yellow', 'white']
    results_df.plot.bar(stacked=True, ax=ax0, color=colors)
    ax0.legend(prop={'size': 10})
    ax0.set_title('bars with legend')
    plt.show()



#Viannou va écrire une super fonction ci-dessous
def w8_loader(model,file_name):
    device = torch.device('cpu')
    dir = os.path.join(os.path.dirname(__file__), f'../weights/{file_name}')
    model.load_state_dict(torch.load(dir, map_location=device))
    return model

def w8_saver(model,file_name):
    dir = os.path.join(os.path.dirname(__file__), f'../weights/{file_name}')
    torch.save(model.net.state_dict(), dir)
    return

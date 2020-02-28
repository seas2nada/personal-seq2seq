import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from utils.directories import CheckDir

import numpy as np

# plot attention graph
def PlotAttention(batch_ys, num_seq, attention_graph, graph_dir, epoch='TEST'):
    fig = plt.figure(figsize=(50,50), dpi=100) # for high resolution, use high dpi
    ax = fig.add_subplot(111)
    ax.matshow(attention_graph.cpu().detach().numpy())
    ax.set_yticklabels(['']+[int(batch_ys[x,-1].cpu().detach().tolist()) for x in range(num_seq)])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(graph_dir+'/epoch_'+str(epoch)+'.png')
    plt.close(fig)
    return

# plot spectrogram with directories
def PlotNpy(spectrogram_dir, feat=None, feat_list=None):
    """

    :param spectrogram_dir: directories for numpy files
    :param feat: single directory
    :param feat_list: list of directory
    """
    if feat_list:
        for feat in feat_list:
            fig = plt.figure() # for high resolution, use high dpi
            ax = fig.add_subplot(111)
            ax.imshow(np.load(feat), origin="lower", aspect="auto", cmap="jet", interpolation="none")
            plt.savefig(spectrogram_dir+'/'+feat.split('/')[-1].split('.npy')[0]+'.png')
            plt.close(fig)
    else:
        fig = plt.figure()  # for high resolution, use high dpi
        ax = fig.add_subplot(111)
        ax.imshow(np.load(feat), origin="lower", aspect="auto", cmap="jet", interpolation="none")
        plt.savefig(spectrogram_dir + feat.split('/')[-1].split('.npy')[0] + '.png')
        plt.close(fig)
    return

# plot spectrograms of batch data
def PlotSignal(batch_spectrogram, iter, save_dir='exp/signal_mel'):
    """

    :param batch_spectrogram: batch array of spectrograms (batch_xs)
    """
    CheckDir([save_dir])
    for i, spectrogram in enumerate(batch_spectrogram):
        fig = plt.figure()  # for high resolution, use high dpi
        ax = fig.add_subplot(111)
        ax.imshow(spectrogram, origin="lower", aspect="auto", cmap="jet", interpolation="none")
        plt.savefig(save_dir + '/' + str(iter) + 'iter_' + str(i))
        plt.close(fig)
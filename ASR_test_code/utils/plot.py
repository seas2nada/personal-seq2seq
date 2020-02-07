import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

import numpy as np

# plot attention graph
def PlotAttention(batch_xs, batch_ys, num_seq, attention_graph, graph_dir, epoch='TEST'):
    fig = plt.figure(figsize=(50,50), dpi=100) # for high resolution, use high dpi
    ax = fig.add_subplot(111)
    ax.matshow(attention_graph.cpu().detach().numpy())
    ax.set_yticklabels(['']+[int(batch_ys[-1,x,-1].cpu().detach().tolist()) for x in range(num_seq)])
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))
    plt.savefig(graph_dir+'/epoch_'+str(epoch)+'.png')
    plt.close(fig)
    return

def PlotSignal(spectrogram_dir, feat=None, feat_list=None):
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
#
# with open('/home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats.scp', 'r') as f:
#     lines = f.readlines()
#     feat_list=[]
#     for i in range(len(lines)):
#         feat_list.append(lines[i].strip('\n'))
#     PlotSignal('./../spectrogram', feat_list=lines)

# a='/home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mjgk-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mjhp-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mjjs2-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mkdb-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mkem-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mmaf-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mmal-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mmap-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mmdg-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mmkw-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mmsh-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mmtm-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mnfe-b.npy \
# /home/dh/Desktop/donghyun/pytorch/ASR_test_code/data/train/feats/cen2-mnjl-b.npy'.split()
# PlotSignal('hi', feat_list=a)
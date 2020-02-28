import torch
from torch.utils.data import DataLoader

from random import randint

import numpy as np
from tqdm import tqdm

sos = 0
eos = 1
ignore_index = -1

# custom dataset generator
class MakeDataset():
    """
    Make batchfied datasets which can be iterated by :parameter index

    :returns Generator-format x, y, x_lens, y_lens data with all batchfied
    """
    def __init__(self, feat, tokens, seqlen, args):
        input_size = args.input_size
        max_in = args.max_in
        max_out = args.max_out

        with open(feat, 'r') as f:
            with open(tokens, 'r') as g:
                with open(seqlen, 'r') as h:
                    feats = f.readlines()
                    indexes = g.readlines()
                    lengths = h.readlines()
                    self.len = len(feats)

                    self.seq_len = torch.zeros(self.len, 1)
                    self.ys_len = torch.zeros(self.len, 1)
                    self.x_data = torch.zeros(self.len, input_size, max_in)
                    self.y_data = torch.full((self.len, max_out+1, 1), eos)
                    self.y_data[:,0,:] = sos

                    for i in range(self.len):
                        self.seq_len[i] = int(lengths[i].strip('\n'))
                        self.x_data[i, :, :int(self.seq_len[i])] = torch.from_numpy(np.load(feats[i].strip('\n')))
                        y = torch.tensor(list(map(int, indexes[i].strip('\n').split()))).unsqueeze(1)
                        self.ys_len[i] = int(y.shape[0])
                        self.y_data[i, 1:int(self.ys_len[i])+1] = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index], self.seq_len[index], self.ys_len[index]

    def __len__(self):
        return self.len

def GetData(feat, tokens, seqlen, args):
    """
    Get batchfied data for iteration training

    :return: loader, contains (batch_xs, batch_ys, sequence_lengths, output_lengths)
    """
    dataset = MakeDataset(feat, tokens, seqlen, args)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=False)

    return loader

def ignore_ys_padded(batch_ys_in, ys_in_len, max_out, device):
    """
    batch_ys_out for gradient descent
    mask, num_ignores for accuracy calculation

    :param batch_ys_in: target batch_ys with <sos> token at front-most, and padded with <eos>
    :param ys_in_len: ys target tokenized sentence lengths
    :param max_out: max output sentence length

    :return: batch_ys_out: target batch_ys without <sos> token, and padded with ignore_index
    :return: ys_mask: mask for ys padded region
    :return: num_ignores: number of ignore_index padded region of decoding_index data
    """
    batch_size = batch_ys_in.shape[0]

    # count number of ignores
    num_ignores = (batch_ys_in == eos).nonzero().shape[0] - batch_size
    # batch_ys_out for loss ignore
    batch_ys_out = batch_ys_in.clone()[:,1:,:] # without <sos>
    # mask for decoder output ignore indexes to be ignored
    ys_mask = torch.ones(batch_size, max_out, 1).to(device)

    # find ignore index for each batch
    # apply it for mask and batch_ys_out
    for batch_iterator in range(batch_size):
        batch_ys_out[batch_iterator, int(ys_in_len[batch_iterator])+1:, :] = ignore_index # ignore after <eos>
        ys_mask[batch_iterator, int(ys_in_len[batch_iterator])+1:, :] = 0 # mask zero after <eos>

    return batch_ys_out, ys_mask, num_ignores

def sort_by_len(batch_xs, batch_ys_in, seqlen, ys_in_len):
    """
    Sort data in descending order for pack_padded_sequence

    :param seqlen: xs frame sequence lengths

    :return: decreasing-order sorted data of: batch_xs, batch_ys_in, seqlen, ys_in_len
    """

    seqlen, sort_idx = seqlen.sort(descending=True)
    batch_xs, batch_ys_in, ys_in_len = batch_xs[sort_idx], batch_ys_in[sort_idx], ys_in_len[sort_idx]

    return batch_xs, batch_ys_in, seqlen, ys_in_len
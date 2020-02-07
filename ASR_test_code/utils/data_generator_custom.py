import torch

from random import randint

import numpy as np
from tqdm import tqdm

# return train/test data loader ( data generator of size 2 [data, target] )
# tuple type data [batch_size, input_size, max_in] / tuple type target [batch_size]
# [(batch size data), (batch size target)], [(batch size data), (batch size target)], ... ]
def GetData(feat, index, args):
    batch_size = args.batch_size
    input_size = args.input_size
    max_in = args.max_in

    with open(feat, 'r') as f:
        with open(index, 'r') as g:
            feats = f.readlines()
            indexes = g.readlines()

            batch_iteration = int(len(feats) / batch_size)
            for i in tqdm(range(batch_iteration)):
                loader = []

                data = np.zeros([batch_size, input_size, max_in])
                target = [0]*batch_size
                for j in range(batch_size):
                    data[j] = np.load(feats[i * batch_size + j].strip('\n'))
                    target[j] = indexes[i * batch_size + j].strip('\n')

                loader.append(tuple(data))
                loader.append(tuple(target))

                yield loader

# Return batched data and target (batch_xs, batch_ys)
# mask, num_ignore for accuracy calculation
# num_seq for attention graph
def BatchData(data, target, batch_size, max_out, sos, eos, ignore_index, device):

    # batch pre-setting (zero-padded)
    batch_ys_in = torch.zeros([batch_size, max_out+1, 1]).to(device)
    batch_ys_out = torch.zeros([batch_size, max_out, 1]).to(device)
    batch_ys_in[:,0,:] = sos

    # count number of ignore indexes to substract from total correct predictions for accuracy calculation
    mask = torch.zeros(batch_size, max_out, 1).to(device) # mask for ignore indexes
    num_ignore = 0 # count number of ignore_index

    # batchfy sequence
    batch_xs = torch.tensor(data).permute(0,2,1).float().to(device)
    for batch_iterator in range(batch_size):

        num_seq = len(target[batch_iterator].split()) # output sequence length
        num_ignore += max_out-(num_seq+1) # add number of ignores for this batch

        # batchfy data, target
        y = torch.tensor(list(map(int, target[batch_iterator].split()))).unsqueeze(1) # target of size [num_seq, 1]

        # insert datas into zero tensors with max_in size
        batch_ys_in[batch_iterator, 1:num_seq+1] = y
        batch_ys_out[batch_iterator, :num_seq] = y
        batch_ys_in[batch_iterator, num_seq+1:] = eos
        batch_ys_out[batch_iterator, num_seq] = eos
        if num_seq != max_out:
            batch_ys_out[batch_iterator, num_seq+1:] = ignore_index

        # mask setting
        mask[batch_iterator, :num_seq+1] = 1

    return batch_xs.to(device), batch_ys_in.to(device), batch_ys_out.to(device), mask.to(device), num_ignore, num_seq
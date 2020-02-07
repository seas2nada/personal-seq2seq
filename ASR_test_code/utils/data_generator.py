import torch
from torch.utils.data import DataLoader

from random import randint

import numpy as np
from tqdm import tqdm

sos = 0
eos = 1

# custom Dataset Maker
class MakeDataset():
    def __init__(self, feat, index, args, device):
        input_size = args.input_size
        max_in = args.max_in
        max_out = args.max_out

        with open(feat, 'r') as f:
            with open(index, 'r') as g:
                feats = f.readlines()
                indexes = g.readlines()
                self.len = len(feats)

                self.x_data = torch.zeros(self.len, input_size, max_in).to(device)
                self.y_data = torch.full((self.len, max_out+1, 1), eos).to(device)
                self.y_data[:,0,:] = sos

                for i in range(self.len):
                    self.x_data[i] = torch.from_numpy(np.load(feats[i].strip('\n')))
                    y = torch.tensor(list(map(int, indexes[i].strip('\n').split()))).unsqueeze(1)
                    self.num_seq = y.shape[0]
                    self.y_data[i, 1:self.num_seq+1] = y

    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    def __len__(self):
        return self.len

def GetData(feat, index, args, device):
    dataset = MakeDataset(feat, index, args, device)
    loader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, drop_last=True)

    return loader

# batch_ys_out for gradient descent
# mask, num_ignores for accuracy calculation
def GetMask(batch_ys_in, batch_size, max_out, device):

    # count number of ignores
    num_ignores = (batch_ys_in == eos).nonzero().shape[0] - batch_size
    # mask for decoder output ignore indexes to be ignored
    mask = torch.ones(batch_size, max_out, 1).to(device)
    # batch_ys_out for loss ignore
    batch_ys_out = torch.zeros(batch_size, max_out, 1).to(device)
    batch_ys_out[:] = batch_ys_in[:,1:,:] # without <sos>

    # find ignore index for each batch
    # apply it for mask and batch_ys_out
    for batch_iterator in range(batch_size):
        # first <eos> appear index in batch_ys_in = second <eos> appear in batch_ys_out
        ignore_index = (batch_ys_in[batch_iterator] == eos).nonzero()[0,0]
        batch_ys_out[batch_iterator, int(ignore_index):, :] = -1 # ignore after <eos>
        mask[batch_iterator, int(ignore_index):, :] = 0 # mask zero after <eos>

    return batch_ys_out, mask, num_ignores
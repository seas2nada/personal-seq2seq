import torch

from random import randint

# Return batched data and target (batch_xs, batch_ys)
def BatchData(data, target, batch_size, max_length, sos, eos, ignore_index, device):

    # batch pre-setting (zero-padded)
    batch_xs = torch.zeros([batch_size, 28*max_length, 28])
    batch_ys_in = torch.zeros([batch_size, max_length+2, 1])
    batch_ys_out = torch.zeros([batch_size, max_length+1, 1])
    batch_ys_in[:,0,:] = sos

    # count number of ignore indexes to substract from total correct predictions for accuracy calculation
    mask = torch.zeros(batch_size, max_length+1, 1).to(device) # mask for ignore indexes
    num_ignore = 0 # count number of ignore_index

    # batchfy sequence
    for batch_iterator in range(batch_size):
        num_seq = randint(1,max_length) # get number of sequence randomly
        num_ignore += max_length-(num_seq+1) # add number of ignores for this batch

        # batchfy data, target
        x = torch.cat([data[j] for j in range(batch_iterator*num_seq, (batch_iterator+1)*num_seq)], 2)
        x = x.transpose(1,2)
        y = target[batch_iterator*num_seq:(batch_iterator+1)*num_seq].view(-1,1)

        # insert datas into zero tensors with max_length size
        batch_xs[batch_iterator, -28*num_seq:] = x  # batch_xs zero-padding should be left-zeros
        batch_ys_in[batch_iterator, 1:num_seq+1] = y
        batch_ys_out[batch_iterator, :num_seq] = y
        batch_ys_in[batch_iterator, num_seq+1:] = eos
        batch_ys_out[batch_iterator, num_seq] = eos
        if num_seq != 10:
            batch_ys_out[batch_iterator, num_seq+1:] = ignore_index

        # mask setting
        mask[batch_iterator, :num_seq+1] = 1

    return batch_xs.to(device), batch_ys_in.to(device), batch_ys_out.to(device), mask.to(device), num_ignore, num_seq

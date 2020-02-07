import torch

# generate mask for test decoder output prediction
# mask after <eos>
# prediction.size() = [batch_size, 1]
def EOSMasking(prediction, eos, device):
    batch_size = prediction.shape[0]
    sequence_len = prediction.shape[1]

    mask = torch.zeros(batch_size, sequence_len, 1)
    eos_indexes = (prediction!=eos).sum(dim=1) # get indexes of eos of size [batch_size]
    for batch_iterator in range(batch_size):
        mask[batch_iterator, :eos_indexes[batch_iterator]+1, :] = 1

    return mask.to(device)

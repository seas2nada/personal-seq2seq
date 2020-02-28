import torch
import math

def to_device(m, x):
    """Send tensor into the device of the module.

    Args:
        m (torch.nn.Module): Torch module.
        x (Tensor): Torch tensor.

    Returns:
        Tensor: Torch tensor located in the same place as torch module.

    """
    assert isinstance(m, torch.nn.Module)
    device = next(m.parameters()).device
    return x.to(device)
"""Initialization functions for RNN sequence-to-sequence models."""

def lecun_normal_init_parameters(module):
    """Initialize parameters in the LeCun's manner."""
    for p in module.parameters():
        data = p.data
        if data.dim() == 1:
            # bias
            data.zero_()
        elif data.dim() == 2:
            # linear weight
            n = data.size(1)
            stdv = 1. / math.sqrt(n)
            data.normal_(0, stdv)
        elif data.dim() in (3, 4):
            # conv weight
            n = data.size(1)
            for k in data.size()[2:]:
                n *= k
            stdv = 1. / math.sqrt(n)
            data.normal_(0, stdv)
        else:
            raise NotImplementedError


def uniform_init_parameters(module):
    """Initialize parameters with an uniform distribution."""
    for p in module.parameters():
        data = p.data
        if data.dim() == 1:
            # bias
            data.uniform_(-0.1, 0.1)
        elif data.dim() == 2:
            # linear weight
            data.uniform_(-0.1, 0.1)
        elif data.dim() in (3, 4):
            # conv weight
            pass  # use the pytorch default
        else:
            raise NotImplementedError


def set_forget_bias_to_one(bias):
    """Initialize a bias vector in the forget gate with one."""
    n = bias.size(0)
    start, end = n // 4, n // 2
    bias.data[start:end].fill_(1.)
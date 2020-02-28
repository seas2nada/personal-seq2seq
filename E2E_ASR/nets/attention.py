import math

import torch
import torch.nn.functional as F

from utils.mask import make_pad_mask
from utils.nets_utils import to_device

# Attention class for Scoring self-attention ex) Add, Dot ...
class Attention(torch.nn.Module):
    """
    Simple scoring attention
    
    :param int hidden_size: # hidden state size of encoder
    :param int emb_size: # units of decoder
    :param int att_size: attention dimension
    """

    def __init__(self, hidden_size, emb_size, att_size, device):
        super(Attention, self).__init__()

        self.mlp_enc = torch.nn.Linear(hidden_size, att_size)
        self.mlp_dec = torch.nn.Linear(emb_size, att_size, bias=False)
        self.gvec = torch.nn.Linear(att_size, 1)

        self.device = device
        self.att_size = att_size
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

        self.device = device
        self.att_size = att_size

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
    
    def forward(self, hidden, encoder_outputs, seqlen, atype, att_prev, scaling=2.0, last_attended_idx=None,
                backward_window=None, forward_window=None):
        batch_size = len(encoder_outputs)

        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None:
            self.enc_h = encoder_outputs
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)
    
        # hidden.size() = [n_layers, batch_size, hidden_size]
        if atype == "AddAtt":
            tan = self.gvec(torch.tanh(self.pre_compute_enc_h + self.mlp_dec(hidden).view(batch_size, 1,
                                                                                          self.att_size)))  # tan.size() = [batch_size, max_in, 1]
            tan = tan.squeeze(2)  # tan.size() = [batch_size, max_in]
        elif atype == "DotAtt":
            tan = torch.sum(
                torch.tanh(self.pre_compute_enc_h) * torch.tanh(self.mlp_dec(hidden)).view(batch_size, 1, self.att_size),
                dim=2)  # tan.size() = [batch_size, max_in]
        else:
            tan = torch.full((batch_size, encoder_outputs.size(1)), 1).to(self.device)

        # mask padded region
        mask = make_pad_mask(seqlen).to(self.device)
        tan.masked_fill_(mask, -float('inf'))

        att_weight = F.softmax(scaling * tan, dim=1)  # att_weight.size() = [batch_size, max_in]

        context_vec = torch.sum(encoder_outputs * att_weight.view(batch_size, att_weight.size(1), 1), dim=1)
        
        return context_vec, att_weight, att_prev

# Attention for location-aware, using convolution layer position-aware
class LocationAttention(torch.nn.Module):
    """
    location-aware attention module.

    Reference: Attention-Based Models for Speech Recognition
        (https://arxiv.org/pdf/1506.07503.pdf)

    :param int hidden_size: # hidden state size
    :param int emb_size: # units of decoder
    :param int att_size: attention dimension
    :param int aconv_chans: # channels of attention convolution
    :param int aconv_filts: filter size of attention convolution
    :param bool han_mode: flag to swith on mode of hierarchical attention and not store pre_compute_enc_h
    """

    def __init__(self, hidden_size, emb_size, att_size, aconv_chans, aconv_filts, device, han_mode=False):
        super(LocationAttention, self).__init__()
        self.mlp_enc = torch.nn.Linear(hidden_size, att_size)
        self.mlp_dec = torch.nn.Linear(emb_size, att_size, bias=False)
        self.mlp_att = torch.nn.Linear(aconv_chans, att_size, bias=False)
        self.loc_conv = torch.nn.Conv2d(
            1, aconv_chans, (1, 2 * aconv_filts + 1), padding=(0, aconv_filts), bias=False)
        self.gvec = torch.nn.Linear(att_size, 1)

        self.device = device
        self.att_size = att_size
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None
        self.han_mode = han_mode

    def reset(self):
        """reset states"""
        self.h_length = None
        self.enc_h = None
        self.pre_compute_enc_h = None
        self.mask = None

    def forward(self, hidden, encoder_outputs, seqlen, atype, att_prev, scaling=2.0, last_attended_idx=None,
                backward_window=1, forward_window=3):
        batch = len(encoder_outputs)

        # pre-compute all h outside the decoder loop
        if self.pre_compute_enc_h is None or self.han_mode:
            self.enc_h = encoder_outputs  # batch_size x frame x hidden_size
            self.h_length = self.enc_h.size(1)
            self.pre_compute_enc_h = self.mlp_enc(self.enc_h)

        # initialize attention weight with uniform dist.
        if att_prev is None:
            # if no bias, 0 0-pad goes 0
            att_prev = (1. - make_pad_mask(seqlen).to(device=self.device, dtype=hidden.dtype))
            whatisthis = seqlen.clone().to(self.device)
            att_prev = att_prev / whatisthis.unsqueeze(-1)

        # att_prev: batch_size x frame -> batch_size x 1 x 1 x frame -> batch_size x att_conv_chans x 1 x frame
        att_conv = self.loc_conv(att_prev.view(batch, 1, 1, self.h_length))
        # att_conv: batch_size x att_conv_chans x 1 x frame -> batch_size x frame x att_conv_chans
        att_conv = att_conv.squeeze(2).transpose(1, 2)
        # att_conv: batch_size x frame x att_conv_chans -> batch_size x frame x att_size
        att_conv = self.mlp_att(att_conv)

        # hidden_tiled: batch_size x frame x att_size
        hidden_tiled = self.mlp_dec(hidden).view(batch, 1, self.att_size)

        # dot with gvec
        # batch_size x frame x att_size -> batch_size x frame
        e = self.gvec(torch.tanh(att_conv + self.pre_compute_enc_h + hidden_tiled)).squeeze(2)

        # NOTE: consider zero padding when compute w.
        if self.mask is None:
            self.mask = to_device(self, make_pad_mask(seqlen))

        e.masked_fill_(self.mask, -float('inf'))

        # apply monotonic attention constraint (mainly for TTS)
        if last_attended_idx is not None:
            e = _apply_attention_constraint(e, last_attended_idx, backward_window, forward_window)

        att_weight = F.softmax(scaling * e, dim=1)

        # weighted sum over flames
        # batch_size x hidden_size
        context_vec = torch.sum(self.enc_h * att_weight.view(batch, self.h_length, 1), dim=1)

        return context_vec, att_weight, att_prev

def initial_att(atype, hidden_size, emb_size, att_size, aconv_chans, aconv_filts, device, han_mode=False):

    if atype == 'LocAtt':
        att = LocationAttention(hidden_size, emb_size, att_size, aconv_chans, aconv_filts, device, han_mode=han_mode)
    else:
        att = Attention(hidden_size, emb_size, att_size, device)

    return att


def attention_for(args, device, han_mode=False):

    # Set attention mode
    att = initial_att(args.atype, args.hidden_size, args.emb_size, args.att_size, args.aconv_chans, args.aconv_filts, device)

    if han_mode:
        att = initial_att(args.atype, args.hidden_size, args.emb_size, args.att_size, args.aconv_chans,
                          args.aconv_filts, device)
        return att

    return att
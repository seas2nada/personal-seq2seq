import torch
import torch.nn.functional as F
import warpctc_pytorch as warp_ctc

import numpy as np

from utils.nets_utils import to_device


class CTC(torch.nn.Module):
    """CTC module
    :param int output_size: dimension of outputs
    :param int hidden_size: number of encoder projection units
    :param float dropout: dropout rate (0.0 ~ 1.0)
    :param str ctc_type: builtin or warpctc
    :param bool reduce: reduce the CTC loss into a scalar
    """

    def __init__(self, output_size, hidden_size, dropout, ctc_type='warpctc', reduce=True):
        super().__init__()
        self.dropout = dropout
        self.loss = None
        self.ctc_lo = torch.nn.Linear(hidden_size, output_size)
        self.ctc_type = ctc_type

        if self.ctc_type == 'builtin':
            reduction_type = 'sum' if reduce else 'none'
            self.ctc_loss = torch.nn.CTCLoss(reduction=reduction_type)
        elif self.ctc_type == 'warpctc':
            self.ctc_loss = warp_ctc.CTCLoss(size_average=True)
        else:
            raise ValueError('ctc_type must be "builtin" or "warpctc": {}'
                             .format(self.ctc_type))

        self.ignore_id = -1
        self.reduce = reduce

    def loss_fn(self, th_pred, th_target, th_ilen, th_olen):
        if self.ctc_type == 'builtin':
            th_pred = th_pred.log_softmax(2)
            # Use the deterministic CuDNN implementation of CTC loss to avoid
            #  [issue#17798](https://github.com/pytorch/pytorch/issues/17798)
            with torch.backends.cudnn.flags(deterministic=True):
                loss = self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
            # Batch-size average
            loss = loss / th_pred.size(1)
            return loss
        elif self.ctc_type == 'warpctc':
            return self.ctc_loss(th_pred, th_target, th_ilen, th_olen)
        else:
            raise NotImplementedError

    def forward(self, encoder_outputs, seqlen, batch_ys_out):
        """CTC forward
        :param torch.Tensor encoder_outputs: batch of padded hidden state sequences (B, Tmax, D)
        :param torch.Tensor seqlen: batch of lengths of hidden state sequences (B)
        :param torch.Tensor batch_ys_out: batch of padded character id sequence tensor (B, Lmax)
        :return: ctc loss value
        :rtype: torch.Tensor
        """
        # TODO(kan-bayashi): need to make more smart way
        ys = [y[y != self.ignore_id] for y in batch_ys_out]  # parse padded ys (delete padded region)

        self.loss = None
        seqlen = torch.from_numpy(np.fromiter(seqlen, dtype=np.int32))
        olens = torch.from_numpy(np.fromiter(
            (x.size(0) for x in ys), dtype=np.int32))

        # zero padding for hs
        ys_hat = self.ctc_lo(F.dropout(encoder_outputs, p=self.dropout))

        # zero padding for ys
        ys_true = torch.cat(ys).cpu().int()  # batch x olen

        # get ctc loss
        # expected shape of seqLength x batchSize x alphabet_size
        dtype = ys_hat.dtype
        ys_hat = ys_hat.transpose(0, 1)
        if self.ctc_type == "warpctc":
            # warpctc only supports float32
            ys_hat = ys_hat.to(dtype=torch.float32)
        else:
            # use GPU when using the cuDNN implementation
            ys_true = to_device(self, ys_true)
        self.loss = to_device(self, self.loss_fn(ys_hat, ys_true, seqlen, olens)).to(dtype=dtype)
        if self.reduce:
            # NOTE: sum() is needed to keep consistency since warpctc return as tensor w/ shape (1,)
            # but builtin return as tensor w/o shape (scalar).
            self.loss = self.loss.sum()

        return self.loss

    def log_softmax(self, encoder_outputs):
        """log_softmax of frame activations
        :param torch.Tensor encoder_outputs: 3d tensor (B, Tmax, hidden_size)
        :return: log softmax applied 3d tensor (B, Tmax, output_size)
        :rtype: torch.Tensor
        """
        return F.log_softmax(self.ctc_lo(encoder_outputs), dim=2)

    def argmax(self, encoder_outputs):
        """argmax of frame activations
        :param torch.Tensor encoder_outputs: 3d tensor (B, Tmax, hidden_size)
        :return: argmax applied 2d tensor (B, Tmax)
        :rtype: torch.Tensor
        """
        return torch.argmax(self.ctc_lo(encoder_outputs), dim=2)


def ctc_for(args, device, reduce=True):
    """Returns the CTC module for the given args and output dimension
    :param Namespace args: the program args
    :param int output_size : The output dimension
    :param bool reduce : return the CTC loss in a scalar
    :return: the corresponding CTC module
    """

    return CTC(args.output_size, args.hidden_size, args.dropout, ctc_type=args.ctc_type, reduce=reduce)

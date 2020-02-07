import torch
from torch import nn
import torch.nn.functional as F

def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states

class RNN(torch.nn.Module):
    """RNN with projection layer module

    :param int idim: dimension of inputs
    :param int elayers: number of encoder layers
    :param int cdim: number of rnn units (resulted in cdim * 2 if bidirectional)
    :param int hdim: number of projection units
    :param np.ndarray subsample: list of subsampling numbers
    :param float dropout: dropout rate
    :param str typ: The RNN type
    """

    def __init__(self, input_size, hidden_size, n_layers, dropout, typ="blstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = torch.nn.LSTM(input_size, hidden_size, n_layers, batch_first=True,
                                   dropout=dropout, bidirectional=bidir)

        if bidir:
            self.l_last = torch.nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.l_last = torch.nn.Linear(hidden_size, hidden_size)
        self.typ = typ
        self.n_layers = n_layers
        self.hidden_size = hidden_size

    def forward(self, input, prev_state=None):
        """RNN forward

        :param torch.Tensor xs_pad: batch of padded input sequences (B, Tmax, D)
        :param torch.Tensor ilens: batch of lengths of input sequences (B)
        :param torch.Tensor prev_state: batch of previous RNN states
        :return: batch of hidden state sequences (B, Tmax, eprojs)
        :rtype: torch.Tensor
        """
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed, it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        output, (hidden, cell) = self.nbrnn(input, hx=prev_state)
        # output: utt list of frame x hidden_size x 2 (2: means bidirectional)

        output = torch.tanh(self.l_last(output))

        return output, hidden, cell

    # initiate hidden
    def initHidden(self, batch_size, device):
        if self.typ[0] == 'b':
            hidden = torch.zeros(self.n_layers*2, batch_size, self.hidden_size, device=device)
            cell = torch.zeros(self.n_layers*2, batch_size, self.hidden_size, device=device)
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            cell = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return hidden, cell
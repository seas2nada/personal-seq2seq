import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

def reset_backward_rnn_state(states):
    """Sets backward BRNN states to zeroes - useful in processing of sliding windows over the inputs"""
    if isinstance(states, (list, tuple)):
        for state in states:
            state[1::2] = 0.
    else:
        states[1::2] = 0.
    return states

class RNN(torch.nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, dropout, device, typ="blstm"):
        super(RNN, self).__init__()
        bidir = typ[0] == "b"
        self.nbrnn = torch.nn.LSTM(input_size, hidden_size, n_layers, batch_first=True,
                                   dropout=dropout, bidirectional=bidir)

        if bidir:
            self.l_last = torch.nn.Linear(hidden_size * 2, hidden_size)
        else:
            self.l_last = torch.nn.Linear(hidden_size, hidden_size)

        self.device = device
        self.typ = typ
        self.n_layers = n_layers
        self.hidden_size = hidden_size

    # initiate hidden
    def initiate_hidden(self, batch_size, device):
        if self.typ[0] == 'b':
            hidden = torch.zeros(2, batch_size, self.hidden_size, device=device)
            cell = torch.zeros(2, batch_size, self.hidden_size, device=device)
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            cell = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

        elayer_states = (hidden, cell)

        return elayer_states

    def forward(self, input, ilens, prev_state=None):
        """
        LSTM Encoder Network Model

        :param input = batch_xs
        :param ilens = seqlen: input frame sequence lengths
        :param prev_state: you can ignore it

        :return: xs_pad: encoder outputs
        :return: ilens: same as input ilens
        :return: elayer_states: you can ignore it
        """

        xs_pack = pack_padded_sequence(input, ilens, batch_first=True)
        self.nbrnn.flatten_parameters()
        if prev_state is not None and self.nbrnn.bidirectional:
            # We assume that when previous state is passed, it means that we're streaming the input
            # and therefore cannot propagate backward BRNN state (otherwise it goes in the wrong direction)
            prev_state = reset_backward_rnn_state(prev_state)
        ys, elayer_states = self.nbrnn(xs_pack, hx=prev_state)
        # ys: utt list of frame x cdim x 2 (2: means bidirectional)
        ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
        # (sum _utt frame_utt) x dim
        projected = torch.tanh(self.l_last(
            ys_pad.contiguous().view(-1, ys_pad.size(2))))
        xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)

        return xs_pad, ilens, elayer_states  # x: utt list of frame x dim

class RNNP(torch.nn.Module):

    def __init__(self, input_size, hidden_size, n_layers, subsample, dropout, device, typ="lstmp"):
        super(RNNP, self).__init__()
        bidir = typ[0] == "b"
        subsample = list(map(int, subsample.split('_')))
        for i in range(n_layers):
            if i == 0:
                inputdim = input_size
            else:
                inputdim = hidden_size
            rnn = torch.nn.LSTM(inputdim, hidden_size, dropout=dropout, num_layers=1, bidirectional=bidir,
                                batch_first=True) if "lstm" in typ \
                else torch.nn.GRU(inputdim, hidden_size, dropout=dropout, num_layers=1, bidirectional=bidir, batch_first=True)
            setattr(self, "%s%d" % ("birnn" if bidir else "rnn", i), rnn)
            # bottleneck layer to merge
            if bidir:
                setattr(self, "bt%d" % i, torch.nn.Linear(2 * hidden_size, hidden_size))
            else:
                setattr(self, "bt%d" % i, torch.nn.Linear(hidden_size, hidden_size))

        self.device = device
        self.n_layers = n_layers
        self.hidden_size = hidden_size
        self.subsample = subsample
        self.typ = typ
        self.bidir = bidir

    # initiate hidden
    def initiate_hidden(self, batch_size, device):
        if self.typ[0] == 'b':
            hidden = torch.zeros(2, batch_size, self.hidden_size, device=device)
            cell = torch.zeros(2, batch_size, self.hidden_size, device=device)
        else:
            hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
            cell = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)

        elayer_states = (hidden, cell)

        return elayer_states

    def forward(self, input, ilens, prev_state=None):
        """
        LSTM Pyramid Encoder Network Model

        :param input = batch_xs
        :param ilens = seqlen: input frame sequence lengths
        :param prev_state: you can ignore it

        :return: xs_pad: encoder outputs
        :return: ilens: sub-sampled encoder outputs lengths
        :return: elayer_states: you can ignore it
        """

        elayer_states = []
        for layer in range(self.n_layers):
            xs_pack = pack_padded_sequence(input, ilens, batch_first=True)
            rnn = getattr(self, ("birnn" if self.bidir else "rnn") + str(layer))
            rnn.flatten_parameters()
            if prev_state is not None and rnn.bidirectional:
                prev_state = reset_backward_rnn_state(prev_state)
            ys, elayer_state = rnn(xs_pack, hx=None if prev_state is None else prev_state)
            elayer_states.append(elayer_state)
            # ys: utt list of frame x cdim x 2 (2: means bidirectional)
            ys_pad, ilens = pad_packed_sequence(ys, batch_first=True)
            sub = self.subsample[layer + 1]
            if sub > 1:
                ys_pad = ys_pad[:, ::sub]
                ilens = torch.tensor([int(i + 1) // sub for i in ilens]).to(self.device)
            # (sum _utt frame_utt) x dim
            projected = getattr(self, 'bt' + str(layer)
                                )(ys_pad.contiguous().view(-1, ys_pad.size(2)))

            if layer == self.n_layers - 1:
                xs_pad = projected.view(ys_pad.size(0), ys_pad.size(1), -1)
            else:
                xs_pad = torch.tanh(projected.view(ys_pad.size(0), ys_pad.size(1), -1))

            input = xs_pad

        return xs_pad, ilens, elayer_states  # x: utt list of frame x dim

def encoding(encoder, seqlen, batch_xs):
    """
    Run Encoder

    :param encoder: encoder model class
    :param seqlen: frame sequence lengths
    :param batch_xs: input frame sequences

    :return: encoder outputs, changed seqlen, states(dummy)
    """
    output, seqlen, states = encoder(batch_xs, seqlen)

    return output, seqlen, states

# set encoder network
def encoder_for(args, device):
    if (args.etype).split('b')[-1]=="lstm":
        return RNN(args.input_size, args.hidden_size, args.elayers, args.dropout, device, typ=args.etype)
    elif (args.etype).split('b')[-1]=="lstmp":
        return RNNP(args.input_size, args.hidden_size, args.elayers, args.subsample, args.dropout, device, typ=args.etype)
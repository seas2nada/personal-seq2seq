import torch
from torch import nn
import torch.nn.functional as F

import random

from utils.mask import make_pad_mask
from utils.nets_utils import to_device
from nets.attention import *

# initiate hidden, cell with size [batch_size * emb_size]
def zero_state(encoder_outputs, hidden_size):
    return encoder_outputs.new_zeros(encoder_outputs.size(0), hidden_size)

# Decoder network
class Decoder(nn.Module):
    def __init__(self, hidden_size, emb_size, output_size, att_size, n_layers, dropout, device):
        super(Decoder, self).__init__()
        self.device = device
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.att_size = att_size
        self.n_layers = n_layers

        self.embedded = nn.Embedding(output_size, emb_size)
        self.dropout = nn.Dropout(dropout)

        self.decoder = torch.nn.ModuleList()
        self.dropout_dec = torch.nn.ModuleList()
        self.decoder += [
            torch.nn.LSTMCell(emb_size + hidden_size, emb_size)]
        self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        for _ in range(1, self.n_layers):
            self.decoder += [
                torch.nn.LSTMCell(emb_size, emb_size)]
            self.dropout_dec += [torch.nn.Dropout(p=dropout)]
        self.out = nn.Linear(emb_size, output_size)

    def reset(self):
        self.enc_h = None
        self.pre_compute_enc_h = None

    def rnn_forward(self, new_input, hidden_list, cell_list, hidden_prev, cell_prev):
        hidden_list[0], cell_list[0] = self.decoder[0](new_input, (hidden_prev[0], cell_prev[0]))
        for l in range(1, self.n_layers):
            hidden_list[l], cell_list[l] = self.decoder[l](
                self.dropout_dec[l - 1](hidden_list[l - 1]), (hidden_prev[l], cell_prev[l]))

        return hidden_list, cell_list

    def forward(self, att, input, hidden_list, cell_list, encoder_outputs, seqlen, atype, att_prev, scaling=2.0, decoding_index=0):
        """

        :param att: attention model class
        :param input: decoder input, which is current target token or previous decoder output
        :param hidden_list: list of hidden states with length of decoding layers
        :param cell_list: list of cell states with length of decoding layers
        :param atype: attention type: LocAtt, AddAtt, DotAtt... etc
        :param att_prev: previous attention weights
        :param scaling: scaling parameter for attention weights
        :param decoding_index: printout decoding for data of which index?

        :return: output: decoder output
        :return: hidden_list, cell_list: decoder output hidden/cell lists
        :return: attention weights for decoding index
        :return: att_prev: current attention will be used for next step
        """

        batch_size = len(encoder_outputs)

        # decoder input embedding
        embedded = self.dropout(self.embedded(input)) # embedded.size() = [batch_size, 1, emb_size]

        # first hidden
        hidden = self.dropout_dec[0](hidden_list[0])
        if hidden is None:
            hidden = encoder_outputs.new_zeros(batch, self.emb_size)
        else:
            hidden = hidden.view(batch_size, self.emb_size)

        context_vec, att_weight, att_prev = att(hidden, encoder_outputs, seqlen, atype, att_prev, scaling=scaling)
        new_input = torch.cat((embedded, context_vec), dim=1)  # attention applied input size = [batch_size, hidden_size*dlayers + emb_size]

        hidden_list, cell_list = self.rnn_forward(new_input, hidden_list, cell_list, hidden_list, cell_list) # pass through LSTM with new input
        output = self.out(self.dropout_dec[-1](hidden_list[-1]))

        # save attention for graph
        attention = att_weight[decoding_index]

        return output, hidden_list, cell_list, attention, att_prev

def decoding(args, decoder, att, batch_ys_in, batch_ys_out, ys_mask, seqlen, encoder_outputs, train):
    """

    :param args: parsed arguments
    :param decoder: decoder model class
    :param att: attention model class
    :param ys_mask: mask for ys padded regions
    :param train: Boolean parameter to decide run-mode: train or test

    :return: decoder_output: decoder outputs, which are hyp
    :return: attention_graph: saved attention weights of decoding_index for plotting graph
    :return: att_loss: loss from attention decoder
    """
    device = decoder.device
    teacher_forcing_ratio = args.teacher_forcing_ratio

    # initiate states
    hidden_list = [zero_state(encoder_outputs, args.emb_size)]
    cell_list = [zero_state(encoder_outputs, args.emb_size)]
    for _ in range(1, args.dlayers):
        cell_list.append(zero_state(encoder_outputs, args.emb_size))
        hidden_list.append(zero_state(encoder_outputs, args.emb_size))
    att.reset() # initialize pre_compute_enc_h in attention
    att_prev = None # initialize previous attention
    decoder_output = []  # save decoded outputs

    batch_size = batch_ys_in.size(0)

    num_seq = int((ys_mask[args.decoding_index] == 0).nonzero()[0, 0]) - 1  # first ignore index-1 = num_seq
    num_x_axis = int(encoder_outputs.size(1) / 1)  # attention graph time sequence x axis tick number
    attention_graph = torch.zeros(num_seq, num_x_axis).to(device)  # save attention for graph

    input = batch_ys_in[:, 0, :].long().squeeze(1)  # first input token should be <sos>
    if train:
        for i in range(args.max_out):
            output, hidden_list, cell_list, attention, att_prev = decoder(att, input, hidden_list, cell_list,
                                                                          encoder_outputs, seqlen, args.atype, att_prev, 
                                                                          decoding_index=args.decoding_index)

            teacher_force = random.random() < teacher_forcing_ratio
            input = batch_ys_in[:, i + 1, :].long().squeeze(1) if (teacher_force and i < args.max_out) else output.argmax(1)  # teacher forcing
            decoder_output.append(output)  # save decoder output for accuracy calculation

            if i < num_seq:
                attention_graph[i] = attention

        decoder_output = torch.stack(decoder_output, dim=1)  # decoder_output.size() = [batch_size, args.max_out, output_size]
        att_loss = F.cross_entropy(decoder_output.contiguous().view(batch_size*args.max_out, -1), batch_ys_out.long().view(-1),
                               ignore_index=-1,
                               reduction='mean')

        return decoder_output, attention_graph, att_loss

    else:
        att_loss = 0 # No loss for testing

        for i in range(args.max_out):
            output, hidden_list, cell_list, attention, att_prev = decoder(att, input, hidden_list, cell_list,
                                                                          encoder_outputs, seqlen, args.atype, att_prev,
                                                                          decoding_index=args.decoding_index)

            input = output.argmax(1)
            decoder_output.append(output)

            if i < num_seq:
                attention_graph[i] = attention  # save current attention to list for graph

        decoder_output = torch.stack(decoder_output, dim=1)  # decoder_output.size() = [batch_size, args.max_out, output_size]

        return decoder_output, attention_graph, att_loss

# set decoder network
def decoder_for(args, device):
    decoder = Decoder(args.hidden_size, args.emb_size, args.output_size, args.att_size, args.dlayers, args.dropout,
                      device)

    return decoder
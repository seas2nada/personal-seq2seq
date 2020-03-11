import torch
from torch import nn
import torch.nn.functional as F

import random
import numpy as np

from utils.mask import make_pad_mask, mask_by_length
from utils.nets_utils import to_device
from nets.attention import *
from nets.ctc_prefix_score import CTCPrefixScoreTH

sos = 0
eos = 1
blank = 2
CTC_SCORING_RATIO = 1.5

# initiate hidden, cell with size [batch_size * emb_size]
def zero_state(encoder_outputs, hidden_size):
    return encoder_outputs.new_zeros(encoder_outputs.size(0), hidden_size)

def end_detect(ended_hyps, i, M=3, D_end=np.log(1 * np.exp(-10))):
    """End detection. """
    if len(ended_hyps) == 0:
        return False
    count = 0
    best_hyp = sorted(ended_hyps, key=lambda x: x['score'], reverse=True)[0]
    for m in range(M):
        # get ended_hyps with their length is i - m
        hyp_length = i - m
        hyps_same_length = [x for x in ended_hyps if len(x['yseq']) == hyp_length]
        if len(hyps_same_length) > 0:
            best_hyp_same_length = sorted(hyps_same_length, key=lambda x: x['score'], reverse=True)[0]
            if best_hyp_same_length['score'] - best_hyp['score'] < D_end:
                count += 1

    if count == M:
        return True
    else:
        return False

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

        self.logzero = -10000000000.0

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

    def recognize_beam_batch(self, att, encoder_outputs, seqlen, lpz, recog_args, normalize_score=True):

        encoder_outputs = mask_by_length(encoder_outputs, seqlen, 0.0)  # encoder outputs padded region zero-maksed

        batch = len(seqlen)
        beam = recog_args.beam_size
        penalty = recog_args.penalty

        ctc_weight = getattr(recog_args, "ctc_weight", 0)
        att_weight = 1.0 - ctc_weight
        ctc_margin = getattr(recog_args, "ctc_window_margin", 0)
        weights_ctc_dec = 1.0
        # weights-ctc, e.g. ctc_loss = w_1*ctc_1_loss + w_2 * ctc_2_loss + w_N * ctc_N_loss

        n_bb = batch * beam
        pad_b = to_device(self, torch.arange(batch) * beam).view(-1, 1)

        ### why max sequence length needed?
        max_seqlen = max(seqlen)
        if recog_args.maxlenratio == 0:
            maxlen = max_seqlen
        else:
            maxlen = max(1, int(recog_args.maxlenratio * max_seqlen))
        # maxlen = recog_args.max_out
        minlen = int(recog_args.minlenratio * max_seqlen)

        # initialization
        cell_prev = [to_device(self, torch.zeros(n_bb, self.emb_size)) for _ in range(self.n_layers)]
        hidden_prev = [to_device(self, torch.zeros(n_bb, self.emb_size)) for _ in range(self.n_layers)]
        cell_list = [to_device(self, torch.zeros(n_bb, self.emb_size)) for _ in range(self.n_layers)]
        hidden_list = [to_device(self, torch.zeros(n_bb, self.emb_size)) for _ in range(self.n_layers)]
        vscores = to_device(self, torch.zeros(batch, beam))
        att_prev = None
        att_w, ctc_scorer, ctc_state = None, None, None
        att.reset()  # reset pre-computation of h

        yseq = [[sos] for _ in range(n_bb)]    # set yseq first token as <sos>
        stop_search = [False]*batch
        ended_hyps = [[] for _ in range(batch)]

        exp_seqlen = seqlen.repeat(beam).view(beam, batch).transpose(0, 1).contiguous()    # repeat seqlen => beam * beam_size
        exp_seqlen = exp_seqlen.view(-1).tolist()   # arange in one list
        exp_encoder_outputs = encoder_outputs.unsqueeze(1).repeat(1, beam, 1, 1).contiguous()   # exp_encoder_outputs.size() = batch * beam_size * max_in * hidden_size
        exp_encoder_outputs = exp_encoder_outputs.view(n_bb, encoder_outputs.size()[1], encoder_outputs.size()[2])  # batch * beam_size => n_bb

        if lpz is not None:
            scoring_ratio = CTC_SCORING_RATIO if att_weight > 0.0 and not lpz.is_cuda else 0
            ctc_scorer = CTCPrefixScoreTH(lpz, seqlen, blank, eos, beam, scoring_ratio, margin=ctc_margin)

        for i in range(maxlen):

            # get context vector and embedded input
            vy = to_device(self, torch.LongTensor(self._get_last_yseq(yseq)))
            ey = self.dropout(self.embedded(vy))
            att_c, att_w, _ = att(self.dropout_dec[0](hidden_prev[0]), exp_encoder_outputs, exp_seqlen, recog_args.atype, att_prev)
            ey = torch.cat((ey, att_c), dim=1)

            # attention decoder
            hidden_list, cell_list = self.rnn_forward(ey, hidden_list, cell_list, hidden_prev, cell_prev)
            logits = self.out(self.dropout_dec[-1](hidden_list[-1]))
            local_scores = att_weight * F.log_softmax(logits, dim=1)

            # ctc
            if ctc_scorer:
                ctc_state, local_ctc_scores = ctc_scorer(yseq, ctc_state, local_scores, att_w)
                local_scores = local_scores + ctc_weight * weights_ctc_dec * local_ctc_scores

            local_scores = local_scores.view(batch, beam, self.output_size)
            if i == 0:
                local_scores[:, 1:, :] = self.logzero

            # accumulate scores
            eos_vscores = local_scores[:, :, eos] + vscores
            vscores = vscores.view(batch, beam, 1).repeat(1, 1, self.output_size)   # batch * beam * output_size
            vscores[:, :, eos] = self.logzero
            vscores = (vscores + local_scores).view(batch, -1)

            # global pruning
            accum_best_scores, accum_best_ids = torch.topk(vscores, beam, 1)    # sort by best score, index
            accum_odim_ids = torch.fmod(accum_best_ids, self.output_size).view(-1).data.cpu().tolist()  # fmod: |
            accum_padded_beam_ids = (torch.div(accum_best_ids, self.output_size) + pad_b).view(-1).data.cpu().tolist() # div: int

            y_prev = yseq[:][:]
            yseq = self._index_select_list(yseq, accum_padded_beam_ids) # get best yseqs
            yseq = self._append_ids(yseq, accum_odim_ids)   # get best next index and append it
            vscores = accum_best_scores
            vidx = to_device(self, torch.LongTensor(accum_padded_beam_ids))

            att_prev = torch.index_select(att_w.view(n_bb, *att_w.shape[1:]), 0, vidx)
            hidden_prev = [torch.index_select(hidden_list[li].view(n_bb, -1), 0, vidx) for li in range(self.n_layers)]
            cell_prev = [torch.index_select(cell_list[li].view(n_bb, -1), 0, vidx) for li in range(self.n_layers)]

            # pick ended hyps
            if i >= minlen:
                k = 0
                penalty_i = (i + 1) * penalty
                thr = accum_best_scores[:, -1]
                for samp_i in range(batch):
                    if stop_search[samp_i]:
                        k = k + beam
                        continue
                    for beam_j in range(beam):
                        _vscore = None
                        if eos_vscores[samp_i, beam_j] > thr[samp_i]:
                            yk = y_prev[k][:]
                            if len(yk) <= seqlen[samp_i]:
                                _vscore = eos_vscores[samp_i][beam_j] + penalty_i
                        elif i == recog_args.max_out - 1:
                            yk = yseq[k][:]
                            _vscore = vscores[samp_i][beam_j] + penalty_i
                        if _vscore:
                            yk.append(eos)
                            _score = _vscore.data.cpu().numpy()
                            ended_hyps[samp_i].append({'yseq': yk, 'vscore': _vscore, 'score': _score})
                        k = k + 1

            # end detection
            stop_search = [stop_search[samp_i] or end_detect(ended_hyps[samp_i], i) for samp_i in range(batch)]
            stop_search_summary = list(set(stop_search))
            if len(stop_search_summary) == 1 and stop_search_summary[0]:
                break

            if ctc_scorer:
                ctc_state = ctc_scorer.index_select_state(ctc_state, accum_best_ids)

        torch.cuda.empty_cache()    ### empty cache?

        dummy_hyps = [{'yseq': [sos, eos], 'score': np.array([-float('inf')])}]
        ended_hyps = [ended_hyps[samp_i] if len(ended_hyps[samp_i]) != 0 else dummy_hyps
                      for samp_i in range(batch)]
        if normalize_score:
            for samp_i in range(batch):
                for x in ended_hyps[samp_i]:
                    x['score'] /= len(x['yseq'])

        nbest_hyps = [sorted(ended_hyps[samp_i], key=lambda x: x['score'],
                             reverse=True)[:min(len(ended_hyps[samp_i]), recog_args.nbest)]
                      for samp_i in range(batch)]

        return nbest_hyps

    @staticmethod
    def _get_last_yseq(exp_yseq):
        last = []
        for y_seq in exp_yseq:
            last.append(y_seq[-1])
        return last

    @staticmethod
    def _append_ids(yseq, ids):
        if isinstance(ids, list):
            for i, j in enumerate(ids):
                yseq[i].append(j)
        else:
            for i in range(len(yseq)):
                yseq[i].append(ids)
        return yseq

    @staticmethod
    def _index_select_list(yseq, lst):
        new_yseq = []
        for l in lst:
            new_yseq.append(yseq[l][:])
        return new_yseq


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
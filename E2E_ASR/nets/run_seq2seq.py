# torch modules
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# basic libraries
import os
import time
import random
from tqdm import tqdm
import numpy as np
import editdistance
from itertools import groupby

# internal modules
import nets.encoders as encoders
import nets.decoders as decoders
import nets.attention as attentions
import nets.ctc as ctcs
from utils.data_generator import ignore_ys_padded, sort_by_len
from utils.directories import CheckDir, ModelDir, FileExists
from utils.mask import make_pad_mask
from utils.plot import PlotAttention, PlotSignal
from utils.nets_utils import *
from utils.index2text import index_to_text

# index setting for sequence
sos = 0
eos = 1
blank = "<blank>"
ignore_index = -1

class Model(nn.Module):
    def __init__(self, args, device):
        super().__init__()

        # model setting
        self.encoder = encoders.encoder_for(args, device)
        self.decoder = decoders.decoder_for(args, device)
        self.att = attentions.attention_for(args, device)
        self.ctc = ctcs.ctc_for(args,device)
        self.device = device
        self.args = args
        self.encoder.to(device)
        self.decoder.to(device)

    def forward(self, batch_xs, seqlen, batch_ys_in, batch_ys_out, ys_mask, train):
        """
        Sequence to Sequence run model: CTC, Attention, Hybrid CTC-Attention
        Only CTC model is not available yet

        :return decoder outputs, attention graph of decoding_index data, total loss
        """
        # 1. RNN encoding
        encoder_outputs, seqlen, states = encoders.encoding(self.encoder, seqlen,
                                                   batch_xs)  # states: [n_layers * (hidden, cell)]

        if self.args.mtl_alpha > 1 or self.args.mtl_alpha < 0:
            raise ValueError('mlt_alpha should be in 0~1. Now: {}'.format(args.mtl_alpha))

        # 2. CTC decoding
        elif self.args.mtl_alpha==1:
            ctc_loss = self.ctc(encoder_outputs, seqlen, batch_ys_out)
            attention_graph = None
            decoder_output = self.ctc.argmax(encoder_outputs).data
            loss = ctc_loss

        # 3. Attention decoding
        elif self.args.mtl_alpha==0:
            decoder_output, attention_graph, att_loss = decoders.decoding(self.args, self.decoder, self.att, batch_ys_in, batch_ys_out,
                                                                  ys_mask, seqlen, encoder_outputs, train=train)
            loss = att_loss

        # 4. Joint CTC-attention decoding
        else:
            ctc_loss = self.ctc(encoder_outputs, seqlen, batch_ys_out)
            decoder_output, attention_graph, att_loss = decoders.decoding(self.args, self.decoder, self.att,
                                                                          batch_ys_in, batch_ys_out,
                                                                          ys_mask, seqlen, encoder_outputs, train=train)
            loss = self.args.mtl_alpha * ctc_loss + (1 - self.args.mtl_alpha) * att_loss

        if math.isnan(float(loss)):
            raise ValueError('loss calculation is incorrect: {}'.format(float(loss)))

        return decoder_output, attention_graph, loss

    def recognize_batch(self, batch_xs, seqlen, recog_args):
        """
        E2E beam search.

        :param batch_xs: batch input feature
        :param seqlen: feature sequence lengths
        :param recog_args: recognize arguments
        :return: y: n_best hyps
        """

        # 1. ENN Encoding
        encoder_outputs, seqlen, states = encoders.encoding(self.encoder, seqlen,
                                                            batch_xs)  # states: [n_layers * (hidden, cell)]

        # calculate log P(z_t|X) for CTC scores
        if recog_args.ctc_weight > 0.0:
            lpz = self.ctc.log_softmax(encoder_outputs)
            normalize_score = False
        else:
            lpz = None
            normalize_score = True

        # 2. Decoder
        seqlen = torch.tensor(list(map(int, seqlen)))  # make sure seqlen is tensor
        y = self.decoder.recognize_beam_batch(self.att, encoder_outputs, seqlen, lpz, recog_args, normalize_score=normalize_score)

        return y

def init_like_chainer(model):
    """Initialize weight like chainer.

    chainer basically uses LeCun way: W ~ Normal(0, fan_in ** -0.5), b = 0
    pytorch basically uses W, b ~ Uniform(-fan_in**-0.5, fan_in**-0.5)
    however, there are two exceptions as far as I know.
    - EmbedID.W ~ Normal(0, 1)
    - LSTM.upward.b[forget_gate_range] = 1 (but not used in NStepLSTM)
    """
    lecun_normal_init_parameters(model)
    # exceptions
    # embed weight ~ Normal(0, 1)
    model.decoder.embedded.weight.data.normal_(0, 1)
    # forget-bias = 1.0
    # https://discuss.pytorch.org/t/set-forget-gate-bias-of-lstm/1745
    for l in range(len(model.decoder.decoder)):
        set_forget_bias_to_one(model.decoder.decoder[l].bias_ih)

def calculate_accuracy(y_hat, y_true, word_eds, word_ref_lens, char_eds, char_ref_lens):
    # calculate sentence distance between hyp and ref
    seq_hat_text = index_to_text([idx for idx in y_hat if int(idx) != -1])
    seq_hat_text = seq_hat_text.replace(blank, '')
    seq_true_text = index_to_text([idx for idx in y_true if int(idx) != -1])
    print(seq_true_text)
    print(seq_hat_text)

    hyp_words = seq_hat_text.split()
    ref_words = seq_true_text.split()
    word_eds.append(editdistance.eval(hyp_words, ref_words))
    word_ref_lens.append(len(ref_words))
    hyp_chars = seq_hat_text.replace(' ', '')
    ref_chars = seq_true_text.replace(' ', '')
    char_eds.append(editdistance.eval(hyp_chars, ref_chars))
    char_ref_lens.append(len(ref_chars))

    return word_eds, word_ref_lens, char_eds, char_ref_lens

def run(train, loader, args, model_dir, device, graph=False, graph_dir=None, model_load_dir=None, decode=False, early_stop=False):

    model = Model(args, device)
    model = model.to(device)

    if model_load_dir!=None:
        # Load saved model
        model.load_state_dict(torch.load(model_load_dir), strict=False)
    else:
        # Initialize parameters with ESPNet style
        init_like_chainer(model)

    if train:
        model.train()
        # Adam optimizer setting
        if args.optimizer=='adam':
            model_optimizer = optim.Adam(model.parameters(), lr=args.learning_rate)
            lr_decay_applied = False

        # Adadelta optimizer setting
        elif args.optimizer=='adadelta':
            model_optimizer = optim.Adadelta(model.parameters(), rho=0.95, eps=args.eps)
            eps_decay_applied = False

        epochs = args.epochs
    else:
        model.eval()
        lr_decay_applied = True
        eps_decay_applied = True

        # epochs = 1 when test
        epochs = 1

    for epoch in range(epochs):
        start_time = time.time()

        # model save directory setting
        model_save_dir = ModelDir(model_dir, epoch, args.learning_rate, args.mtl_alpha)

        iter = 0  # iteration count
        total_wer = 0  # batch total WER
        total_cer = 0   # batch total CER
        total_loss = 0  # batch total loss
        for data, target, seq_len, ys_len in tqdm(loader):

            batch_xs = data.permute(0, 2, 1).to(device)
            batch_ys_in = target.to(device)
            seqlen = seq_len.squeeze(1).to(device)
            ys_in_len = ys_len.squeeze(1).to(device)

            # sort by decreasing order for pack_padded_sequence
            batch_xs, batch_ys_in, seqlen, ys_in_len = sort_by_len(batch_xs, batch_ys_in, seqlen, ys_in_len)

            # get mask for accuracy calculation & ys input sequence length
            batch_ys_out, ys_mask, num_ignores = ignore_ys_padded(batch_ys_in, ys_in_len, args.max_out, device)

            # run encoder-decoder seq2seq model
            decoder_output, attention_graph, loss = model(batch_xs, seqlen, batch_ys_in, batch_ys_out, ys_mask, train)

            # output number of sequence for printout decoding result and attention graph of decoding_index data
            num_seq = int(ys_in_len[args.decoding_index])

            # model update
            if train:
                loss *= (np.mean([int(x) for x in ys_in_len]) - 1)  # loss normalize
                loss.backward()  # back propagation
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_threshold)  # clipping
                model_optimizer.step()  # forward propagation

                # save models
                torch.save(model.state_dict(), model_save_dir)
                model_optimizer.zero_grad()  # zero_grad

                # loss calculation
                total_loss += loss.item()
                avg_loss = total_loss / (iter + 1)

            # accuracy calculation
            prediction = decoder_output  # prediction.size() = [batch_size, args.max_out, 1]
            word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []

            # attention based decoding (Attention or Joint CTC-attention)
            if args.mtl_alpha<1:
                prediction = prediction.argmax(2).unsqueeze(2)
                prediction.masked_fill_(~ys_mask.bool(), -1)

                for i, y_hat in enumerate(prediction):
                    y_true = batch_ys_out[i]
                    word_eds, word_ref_lens, char_eds, char_ref_lens = calculate_accuracy(y_hat, y_true, word_eds,
                                                                                          word_ref_lens, char_eds,
                                                                                          char_ref_lens)
                    if i==args.decoding_index:
                        seq_hat_text_print = index_to_text([idx for idx in y_hat if int(idx) != -1]).replace(blank, '')
                        seq_true_text_print = index_to_text([idx for idx in y_true if int(idx) != -1])

            # CTC based decoding (Character-based CTC)
            else:
                for i, y in enumerate(prediction):
                    y_hat = [x[0] for x in groupby(y)]
                    y_true = batch_ys_out[i]
                    word_eds, word_ref_lens, char_eds, char_ref_lens = calculate_accuracy(y_hat, y_true, word_eds,
                                                                                          word_ref_lens, char_eds,
                                                                                          char_ref_lens)
                    if i==args.decoding_index:
                        seq_hat_text_print = index_to_text([idx for idx in y_hat if int(idx) != -1]).replace(blank, '')
                        seq_true_text_print = index_to_text([idx for idx in y_true if int(idx) != -1])

            # calculate average CER
            cer = float(sum(char_eds))*100 / sum(char_ref_lens)
            wer = float(sum(word_eds))*100 / sum(word_ref_lens)
            total_cer += cer
            total_wer += wer
            avg_cer = total_cer / (iter+1)
            avg_wer = total_wer / (iter + 1)
            iter+=1

        # Apply learning rate or epsilon decay
        if args.optimizer == "adam" and args.lr_decay is not None and not lr_decay_applied and float(avg_loss) < 5:
            for p in model_optimizer.param_groups:
                p["lr"] *= args.lr_decay
            lr_decay_applied = True
            print("Learning rate has been decayed to {}".format(args.learning_rate * args.eps_decay))
        elif args.optimizer == "adadelta" and args.eps_decay is not None and not eps_decay_applied and float(avg_loss) < 5:
            for p in model_optimizer.param_groups:
                p["eps"] *= args.eps_decay
            eps_decay_applied = True
            print("Epsilon has been decayed to {}".format(args.eps * args.eps_decay))

        # plot attention graph
        if graph:
            if args.mtl_alpha==1:
                raise ValueError('graph should be false when mtl_alpha=1 (CTC-only)')
            file_prefix=epoch if train else "TEST"
            PlotAttention(batch_ys_out[args.decoding_index], num_seq, attention_graph, graph_dir, file_prefix)

        # printout decoding result without beam search
        end_time = time.time()
        if decode:
            print('hyp:', seq_hat_text_print)
            print('ref:', seq_true_text_print)
        if train:
            print('Epoch: {}\tLoss: {:.3f}\tCER: {:.3f}\tWER: {:.3f}'.format(epoch + 1, avg_loss, avg_cer, avg_wer))
        else:
            print('Test CER: {:.3f}\tWER: {:.3f}'.format(avg_cer, avg_wer))
        print('Elapsed Time: {}s'.format(int(end_time - start_time)))

        # stop early when loss decreased enough
        if train and early_stop and float(avg_cer)<1:
            print("Training has been stopped early with epoch: {}".format(epoch+1) + '\n')
            break

    return epoch+1

def recog(loader, recog_args, model_load_dir, result_log, device):
    start_time = time.time()

    model = Model(recog_args, device)
    model = model.to(device)

    # Load saved model
    model.load_state_dict(torch.load(model_load_dir), strict=False)
    model.eval()

    # result log
    if FileExists(result_log):
        os.remove(result_log)
    f = open(result_log, 'a')

    iter = 0
    total_cer = 0
    total_wer = 0

    for data, target, seq_len, ys_len in tqdm(loader):

        batch_xs = data.permute(0, 2, 1).to(device)
        batch_ys_in = target.to(device)
        seqlen = seq_len.squeeze(1).to(device)
        ys_in_len = ys_len.squeeze(1).to(device)

        # sort by decreasing order for pack_padded_sequence
        batch_xs, batch_ys_in, seqlen, ys_in_len = sort_by_len(batch_xs, batch_ys_in, seqlen, ys_in_len)

        # get mask for accuracy calculation & ys input sequence length
        batch_ys_out, ys_mask, num_ignores = ignore_ys_padded(batch_ys_in, ys_in_len, recog_args.max_out, device)

        word_eds, word_ref_lens, char_eds, char_ref_lens = [], [], [], []

        with torch.no_grad():
            nbest_hyps = model.recognize_batch(batch_xs, seqlen, recog_args)

            for i in range(len(nbest_hyps)):
                y_hat = nbest_hyps[i][0]['yseq'][1:-1]
                y_true = batch_ys_out[i, :int(ys_in_len[i])]
                word_eds, word_ref_lens, char_eds, char_ref_lens = calculate_accuracy(y_hat, y_true, word_eds,
                                                                                      word_ref_lens, char_eds,
                                                                                      char_ref_lens)
                f.write("hyp:\t" + index_to_text(nbest_hyps[i][0]['yseq'][1:-1]) + '\n')
                f.write("ref:\t" + index_to_text(batch_ys_out[i, :int(ys_in_len[i])]) + '\n' + '\n')

            cer = float(sum(char_eds)) * 100 / sum(char_ref_lens)
            wer = float(sum(word_eds)) * 100 / sum(word_ref_lens)
            total_cer += cer
            total_wer += wer
            avg_cer = total_cer / (iter + 1)
            avg_wer = total_wer / (iter + 1)
            iter += 1

    end_time = time.time()
    f.write("Avg CER: {:.3f}\tAvg WER: {:.3f}".format(avg_cer, avg_wer))
    print('Decoding CER: {:.3f}\tWER: {:.3f}'.format(avg_cer, avg_wer))
    print('Elapsed Time: {}s'.format(int(end_time - start_time)))
    f.close()
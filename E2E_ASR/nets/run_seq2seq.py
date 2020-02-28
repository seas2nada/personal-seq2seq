# torch modules
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# basic libraries
import time
import random
from tqdm import tqdm
import numpy as np

# internal modules
import nets.encoders as encoders
import nets.decoders as decoders
import nets.attention as attentions
import nets.ctc as ctcs
from utils.data_generator import ignore_ys_padded, sort_by_len
from utils.directories import CheckDir, ModelDir
from utils.mask import make_pad_mask
from utils.plot import PlotAttention, PlotSignal
from utils.nets_utils import *
from utils.index2text import index_to_text

# index setting for sequence
sos = 0
eos = 1
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
        # RNN encoding
        encoder_outputs, seqlen, states = encoders.encoding(self.encoder, seqlen,
                                                   batch_xs)  # states: [n_layers * (hidden, cell)]

        # Attention based decoding
        decoder_output, attention_graph, att_loss = decoders.decoding(self.args, self.decoder, self.att, batch_ys_in, batch_ys_out,
                                                                  ys_mask, seqlen, encoder_outputs, train=train)

        if self.args.mtl_alpha > 1 or self.args.mtl_alpha < 0:
            raise ValueError('mlt_alpha should be in 0~1. Now: {}'.format(args.mtl_alpha))
        # Use hybrid CTC-attention
        elif self.args.mtl_alpha > 0 and self.args.mtl_alpha < 1:
            ctc_loss = self.ctc(encoder_outputs, seqlen, batch_ys_out)
            loss = self.args.mtl_alpha * ctc_loss + (1 - self.args.mtl_alpha) * att_loss
        # Use only attention
        else:
            loss = att_loss
        if math.isnan(float(loss)):
            raise ValueError('loss calculation is incorrect: {}'.format(float(loss)))

        return decoder_output, attention_graph, loss

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
        model_save_dir = ModelDir(model_dir, epoch, args.learning_rate)

        iter = 0  # iteration count
        total_acc = 0  # batch total accuracy
        total_loss = 0  # batch total loss
        for data, target, seq_len, ys_len in tqdm(loader):
            batch_size = data.size(0)

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
            prediction = decoder_output.argmax(2).unsqueeze(2)  # prediction.size() = [batch_size, args.max_out, 1]
            prediction = prediction * ys_mask  # mask after <eos>
            answer = batch_ys_out * ys_mask    # mask after <eos>

            accuracy = (prediction.eq(answer).sum() - num_ignores) * 100 / (
                        args.max_out * batch_size - num_ignores)  # accuracy calculation excluding ignore indexes
            total_acc += accuracy
            avg_acc = total_acc / (iter + 1)

            iter += 1

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
            file_prefix=epoch if train else "TEST"
            PlotAttention(batch_ys_out[args.decoding_index], num_seq, attention_graph, graph_dir, file_prefix)

        # printout decoding result without beam search
        end_time = time.time()
        if decode:
            print('hyp:', index_to_text(prediction[args.decoding_index, :num_seq + 1].squeeze(1)))
            print('ref:', index_to_text(batch_ys_out[args.decoding_index, :num_seq + 1].squeeze(1)))
        if train:
            print('Epoch: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch + 1, avg_loss, avg_acc))
        else:
            print('Test Accuracy: {:.3f}'.format(avg_acc))
        print('Elapsed Time: {}s'.format(int(end_time - start_time)))

        # stop early when loss decreased enough
        if train and early_stop and float(avg_loss)<1:
            print("Training has been stopped early with epoch: {}".format(epoch+1) + '\n')
            break

    return epoch+1
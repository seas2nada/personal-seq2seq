# torch modules
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# basic libraries
import time
import random
from tqdm import tqdm

# internal modules
import nets.encoders as encoders
import nets.decoders as decoders
from utils.data_generator import GetMask
from utils.directories import CheckDir, ModelDir
from utils.mask import EOSMasking
from utils.plot import PlotAttention

# index setting for sequence
sos = 0
eos = 1
ignore_index = -1

def train(train_loader, args, model_save_dir, graph_dir, device, enc_load_dir=None, dec_load_dir=None, decode=False, load=False):

    # model setting
    encoder = encoders.RNN(args.input_size, args.hidden_size, args.elayers, args.dropout, typ='lstm')
    # decoder = decoders.Decoder(args.hidden_size, args.emb_size, args.output_size, args.dlayers, args.dropout, device)
    decoder = decoders.AttentionDecoder(args.batch_size, args.hidden_size, args.emb_size, args.output_size, args.dlayers, args.dropout, device)

    if load:
        encoder.load_state_dict(torch.load(enc_load_dir))
        decoder.load_state_dict(torch.load(dec_load_dir))
    encoder.to(device)
    decoder.to(device)
    encoder.train()
    decoder.train()

    # optimizer setting
    encoder_optimizer = optim.Adam(encoder.parameters(), lr=args.learning_rate)
    decoder_optimizer = optim.Adam(decoder.parameters(), lr=args.learning_rate)

    # encoder_optimizer = optim.Adadelta(encoder.parameters())
    # decoder_optimizer = optim.Adadelta(decoder.parameters())

    for epoch in range(args.epochs):
        start_time = time.time()

        # model save directory setting
        enc_save_dir, dec_save_dir = ModelDir(model_save_dir, epoch, args.learning_rate)

        k=0 # iteration count
        total_acc = 0 # batch total accuracy
        total_loss = 0 # batch total loss
        for data, target in tqdm(train_loader):
            loss = 0

            batch_xs = data.permute(0,2,1)
            batch_ys_in = target

            # get mask for accuracy calculation
            batch_ys_out, mask, num_ignores = GetMask(batch_ys_in, args.batch_size, args.max_out, device)

            encoder_optimizer.zero_grad() # set optimizer
            decoder_optimizer.zero_grad() # set optimizer

            # RNN encoding
            hidden, cell = encoder.initHidden(args.batch_size, device)
            output, hidden, cell = encoder(batch_xs, prev_state=(hidden, cell))
            hidden, cell = hidden[1].unsqueeze(0), cell[1].unsqueeze(0)
            encoder_outputs = output.permute(1,0,2)

            # Attention based decoding
            decoder_output = torch.zeros(args.max_out, args.batch_size, args.output_size).to(device)
            input = batch_ys_in[:,0,:].long()

            num_seq = int((mask[-1]==0).nonzero()[0,0])-1 # ignore_index-1 = num_seq
            num_x_axis = 100 # attention graph time sequence x axis tick number
            attention_graph = torch.zeros(num_seq, num_x_axis).to(device) # save attention for graph
            for i in range(args.max_out):
                output, hidden, cell, attention = decoder(input, hidden, cell, encoder_outputs)
                teacher_force = random.random() < args.teacher_forcing_ratio
                input = batch_ys_in[:,i+1,:].long() if (teacher_force and i < args.max_out) else output.argmax(2) # teacher forcing
                decoder_output[i] = output.squeeze(1) # save decoder output for accuracy calculation
                loss += F.cross_entropy(output.squeeze(1), batch_ys_out[:,i].long().squeeze(1), ignore_index=-1, reduction='mean')

                # save current attention to list for graph
                if i < num_seq:
                    dc_attention = torch.zeros(num_x_axis)
                    for step in range(num_x_axis):
                        dc_attention[step] = attention[step*int(args.max_in/num_x_axis):(step+1)*int(args.max_in/num_x_axis)].sum()
                    attention_graph[i] = dc_attention/num_x_axis

            loss.backward() # back propagation
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_threshold) # clipping
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_threshold) # clipping

            encoder_optimizer.step() # forward propagation
            decoder_optimizer.step() # forward propagation

            decoder_output = decoder_output.transpose(0,1) # decoder_output.size() = [batch_size, args.max_out, output_size]
            prediction = decoder_output.argmax(2).unsqueeze(2) # prediction.size() = [batch_size, args.max_out, 1]

            # mask after <eos>
            prediction = prediction*mask
            answer = batch_ys_out*mask

            accuracy = (prediction.eq(answer).sum()-num_ignores)*100/(args.max_out*args.batch_size-num_ignores) # accuracy calculation excluding ignore index
            total_acc += accuracy
            avg_acc = total_acc/(k+1)
            total_loss += loss.item()
            avg_loss = total_loss/(k+1)
            k+=1

        # save models
        torch.save(encoder.state_dict(), enc_save_dir)
        torch.save(decoder.state_dict(), dec_save_dir)

        # plot attention graph
        PlotAttention(batch_xs, batch_ys_out, num_seq, attention_graph, graph_dir, epoch)

        # printout result
        end_time = time.time()
        if decode: ### need to be changed
            print('Predict:', prediction[-1, :num_seq+1].transpose(0,1))
            print('Target:', batch_ys_out[-1,:num_seq+1].transpose(0,1))
        print('Epoch: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch+1, avg_loss, avg_acc))
        print('Elapsed Time: {}s'.format(int(end_time-start_time)))

    return

def test(test_loader, args, enc_load_dir, dec_load_dir, device, graph=False, graph_dir=None, decode=True, load=True):

    # model setting
    encoder = encoders.RNN(args.input_size, args.hidden_size, args.elayers, args.dropout, typ='lstm')
    decoder = decoders.AttentionDecoder(args.batch_size, args.hidden_size, args.emb_size, args.output_size, args.dlayers, args.dropout, device)

    if load:
        encoder.load_state_dict(torch.load(enc_load_dir))
        decoder.load_state_dict(torch.load(dec_load_dir))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    k=0 # iteration count
    total_acc = 0 # batch total accuracy
    for data, target in tqdm(test_loader):

        batch_xs = data.permute(0,2,1)
        batch_ys_in = target

        # get mask for accuracy calculation
        batch_ys_out, mask, num_ignores = GetMask(batch_ys_in, args.batch_size, args.max_out, device)

        # RNN encoding
        hidden, cell = encoder.initHidden(args.batch_size, device)

        output, hidden, cell = encoder(batch_xs, prev_state=(hidden, cell))
        hidden, cell = hidden[1].unsqueeze(0), cell[1].unsqueeze(0)
        encoder_outputs = output.permute(1, 0, 2)

        # Attention based decoding
        decoder_output = torch.zeros(args.max_out, args.batch_size, args.output_size).to(device)
        input = batch_ys_in[:,0,:].long()

        num_seq = int((mask[-1]==0).nonzero()[0,0])-1 # ignore_index-1 = num_seq
        num_x_axis = 100 # attention graph time sequence x axis tick number
        attention_graph = torch.zeros(num_seq, num_x_axis).to(device) # save attention for graph
        for i in range(args.max_out):
            output, hidden, cell, attention = decoder(input, hidden, cell, encoder_outputs)
            input = output.argmax(2) # teacher forcing
            decoder_output[i] = output.squeeze(1) # save decoder output for accuracy calculation

            # save current attention to list for graph
            if i < num_seq:
                dc_attention = torch.zeros(num_x_axis)
                for step in range(num_x_axis):
                    dc_attention[step] = attention[step*int(args.max_in/num_x_axis):(step+1)*int(args.max_in/num_x_axis)].sum()
                attention_graph[i] = dc_attention/num_x_axis

        decoder_output = decoder_output.transpose(0,1) # decoder_output.size() = [batch_size, args.max_out, output_size]
        prediction = decoder_output.argmax(2).unsqueeze(2) # prediction.size() = [batch_size, args.max_out, 1]

        # mask after <eos>
        prediction = prediction*mask
        answer = batch_ys_out*mask

        accuracy = (prediction.eq(answer).sum()-num_ignores)*100/(args.max_out*args.batch_size-num_ignores) # accuracy calculation excluding ignore index
        total_acc += accuracy
        avg_acc = total_acc/(k+1)
        k+=1

    # plot attention graph
    if graph:
        PlotAttention(batch_xs, batch_ys_out, num_seq, attention_graph, graph_dir, 'TEST')

    # printout result
    if decode: ### need to be changed
        print('Predict:', prediction[-1, :num_seq+1].transpose(0,1))
        print('Target:', batch_ys_out[-1,:num_seq+1].transpose(0,1))
    print('Accuracy: {:.3f}'.format(avg_acc))

    return
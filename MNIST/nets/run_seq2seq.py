# torch modules
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# basic libraries
import time
import random

# internal modules
import nets.encoders as encoders
import nets.decoders as decoders
from utils.directories import CheckDir, ModelDir
from utils.batchfy import BatchData
from utils.mask import EOSMasking
from utils.plot import PlotAttention

# index setting for sequence
sos = 10
eos = 11
ignore_index = -1

def train(train_loader, args, model_save_dir, graph_dir, device, enc_load_dir=None, dec_load_dir=None, decode=False, load=False):

    # model setting
    encoder = encoders.RNN(args.input_size, args.hidden_size, args.n_layers, args.dropout, device)
    decoder = decoders.AttentionDecoder(args.batch_size, args.hidden_size, args.emb_size, args.output_size, args.n_layers, args.dropout, device)
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

    for epoch in range(args.epochs):
        start_time = time.time()

        # model save directory setting
        enc_save_dir, dec_save_dir = ModelDir(model_save_dir, epoch, args.learning_rate)

        k=0 # iteration count
        total_acc = 0 # batch total accuracy
        total_loss = 0 # batch total loss
        for data, target in train_loader:
            loss = 0

            # batchfy data
            batch_xs, batch_ys_in, batch_ys_out, mask, num_ignore, num_seq = \
            BatchData(data, target, args.batch_size, args.max_length, sos, eos, ignore_index, device)

            encoder_optimizer.zero_grad() # set optimizer
            decoder_optimizer.zero_grad() # set optimizer

            # RNN encoding
            encoder_outputs = torch.zeros(args.max_length*28, args.batch_size, args.hidden_size).to(device)
            hidden, cell = encoder.initHidden(args.batch_size, device)
            for i in range(args.max_length*28):
                output, hidden, cell = encoder(batch_xs[:,i,:].unsqueeze(1), hidden, cell)
                encoder_outputs[i] = output.squeeze(1) # output.size() = [batch_size, 1, hidden_size]

            # Attention based decoding
            decoder_output = torch.zeros(args.max_length+1, args.batch_size, args.output_size).to(device)
            input = batch_ys_in[:,0,:].long()

            attention_graph = torch.zeros(num_seq, num_seq*28).to(device) # save attention for graph
            for i in range(args.max_length+1):
                output, hidden, cell, attention = decoder(input, hidden, cell, encoder_outputs)
                teacher_force = random.random() < args.teacher_forcing_ratio
                input = batch_ys_in[:,i+1,:].long() if (teacher_force and i < args.max_length+1) else output.argmax(2) # teacher forcing
                decoder_output[i] = output.squeeze(1) # save decoder output for accuracy calculation
                loss += F.cross_entropy(output.squeeze(1), batch_ys_out[:,i].long().squeeze(1), ignore_index=-1, reduction='mean')

                # save current attention to list for graph
                if i < num_seq:
                    attention_graph[i] = attention[-num_seq*28:]

            loss.backward() # back propagation
            torch.nn.utils.clip_grad_norm_(encoder.parameters(), args.clip_threshold) # clipping
            torch.nn.utils.clip_grad_norm_(decoder.parameters(), args.clip_threshold) # clipping

            encoder_optimizer.step() # forward propagation
            decoder_optimizer.step() # forward propagation

            decoder_output = decoder_output.transpose(0,1) # decoder_output.size() = [batch_size, args.max_length, output_size]
            prediction = decoder_output.argmax(2).unsqueeze(2) # prediction.size() = [batch_size, args.max_length, 1]

            # mask after <eos>
            prediction = prediction*mask
            answer = batch_ys_out*mask

            accuracy = (prediction.eq(answer).sum()-num_ignore)*100/((args.max_length+1)*args.batch_size-num_ignore) # accuracy calculation excluding ignore index
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
        if decode:
            print('Predict:', prediction[-1, :num_seq+1].transpose(0,1))
            print('Target:', batch_ys_out[-1,:num_seq+1].transpose(0,1))
        print('Epoch: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch, avg_loss, avg_acc))
        print('Elapsed Time: {}s'.format(int(end_time-start_time)))

    return

def test(test_loader, args, enc_load_dir, dec_load_dir, device, graph=True, decode=True, load=True):

    # model setting
    encoder = encoders.RNN(args.input_size, args.hidden_size, args.n_layers, args.dropout, device)
    decoder = decoders.AttentionDecoder(args.batch_size, args.hidden_size, args.emb_size, args.output_size, args.n_layers, args.dropout, device)
    if load:
        encoder.load_state_dict(torch.load(enc_load_dir))
        decoder.load_state_dict(torch.load(dec_load_dir))
    encoder.to(device)
    decoder.to(device)
    encoder.eval()
    decoder.eval()

    start_time = time.time()

    k=0 # iteration count
    total_acc = 0 # batch total accuracy
    for data, target in test_loader:

        # batchfy data
        batch_xs, batch_ys_in, batch_ys_out, mask, num_ignore, num_seq = \
        BatchData(data, target, args.batch_size, args.max_length, sos, eos, ignore_index, device)

        # RNN encoding
        encoder_outputs = torch.zeros(args.max_length*28, args.batch_size, args.hidden_size).to(device)
        hidden, cell = encoder.initHidden(args.batch_size, device)
        for i in range(args.max_length*28):
            output, hidden, cell = encoder(batch_xs[:,i,:].unsqueeze(1), hidden, cell)
            encoder_outputs[i] = output.squeeze(1) # output.size() = [batch_size, 1, hidden_size]

        # Attention based decoding
        decoder_output = torch.zeros(args.max_length+1, args.batch_size, args.output_size).to(device)
        input = torch.full((args.batch_size, 1), sos).to(device).long()

        attention_graph = torch.zeros(num_seq, num_seq*28).to(device) # save attention for graph
        for i in range(args.max_length+1):
            output, hidden, cell, attention = decoder(input, hidden, cell, encoder_outputs)
            input = output.argmax(2)
            decoder_output[i] = output.squeeze(1) # save decoder output for accuracy calculation

            # save current attention to list for graph
            if i < num_seq:
                attention_graph[i] = attention[-num_seq*28:]

        decoder_output = decoder_output.transpose(0,1) # decoder_output.size() = [batch_size, args.max_length, output_size]
        prediction = decoder_output.argmax(2).unsqueeze(2) # prediction.size() = [batch_size, args.max_length, 1]

        # mask after <eos>
        mask = EOSMasking(prediction, eos, device)
        prediction = prediction*mask
        answer = batch_ys_out*mask

        accuracy = (prediction.eq(answer).sum()-num_ignore)*100/((args.max_length+1)*args.batch_size-num_ignore) # accuracy calculation excluding ignore index
        total_acc += accuracy
        avg_acc = total_acc/(k+1)
        k+=1

    # plot attention graph
    if graph:
        graph_dir = './test_graph'
        CheckDir([graph_dir])
        PlotAttention(batch_xs, batch_ys_out, num_seq, attention_graph, graph_dir)

    # printout result
    end_time = time.time()
    if decode:
        print('Predict:', prediction[-1, :num_seq+1].transpose(0,1))
        print('Target:', batch_ys_out[-1,:num_seq+1].transpose(0,1))
    print('Test Accuracy: {:.3f}'.format(avg_acc))
    print('Elapsed Time: {}s'.format(int(end_time-start_time)))

    return

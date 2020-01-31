### embedding matrix trained together with token level

import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

import numpy as np
import matplotlib.pyplot as plt

import time
import random
from random import randint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Encoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(Encoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return hidden, cell

    def initHidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return hidden, cell

class Decoder(nn.Module):
    def __init__(self, hidden_size, emb_size, output_size, n_layers, dropout):
        super(Decoder, self).__init__()
        self.output_size = output_size
        self.hidden_size = hidden_size
        self.embedded = nn.Embedding(output_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.out(output)
        return output, hidden, cell

class CNNaccum(nn.Module):
    # CNN accumulator to summarize decoder input
    def __init__(self, output_size, emb_size, hidden_size, num_seq, kernel_size=3, n_layers=3, accum_param=1, exp_size=10, vertical=False):
        super(CNNaccum, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.kernel_size = kernel_size
        self.emb2hid = nn.Linear(emb_size, hidden_size)
        self.exp_size = 28 if vertical else exp_size
        self.expand = nn.Linear(1, self.exp_size)
        self.output_size = output_size*accum_param if vertical else 28*num_seq
        self.fc_out = nn.Linear(hidden_size, self.output_size)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hidden_size, 
                                              out_channels = 2 * hidden_size, 
                                              kernel_size = kernel_size)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        batch_size = input.shape[0]
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.permute(0, 2, 1) # conv_input = [batch size, hidden_size, 1]
        conv_input = self.expand(conv_input) # conv_input = [batch size, hidden_size, output_size]
        
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]

        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)

            # zero padding
            padding = torch.zeros(batch_size, 
                                  hidden_size, 
                                  self.kernel_size - 1).fill_(0).to(device)
            padded_conv_input = torch.cat((padding, conv_input), dim = 2) #padded_conv_input = [batch size, hidden_size, num_seq+2 + kernel size - 1]
        
            # pass through convolutional layer
            conved = conv(padded_conv_input) #conved = [batch size, 2 * hidden_size, num_seq+2]
            conved = F.glu(conved, dim = 1) #conved = [batch size, hidden_size, num_seq+2]

            # apply residual connection
            conved = (conved + conv_input) * self.scale
            conv_input = conved
            
        output = self.fc_out(self.dropout(conved.permute(0, 2, 1))) #output = [batch size, self.output_size, self.exp_size]
        return output.permute(0,2,1)

class Embed(nn.Module):
    # Encoder-Decoder shared embedding layer
    def __init__(self, output_size, emb_size, dropout):
        super(Embed, self).__init__()
        self.embedding = nn.Embedding(output_size, emb_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input):
        embedded = self.dropout(self.embedding(input))
        return embedded

vertical = False # decide to concat horizontally or vertically, default = False
num_seq = 8
accum_param=1
batch_size=100
MAXLENGTH=10
epochs=200
exp_size = 10   # expand one token to exp_size, default = 10
input_size = 28 if vertical else 28+exp_size
output_size = 12
emb_size = 128
hidden_size = 256
learning_rate = 0.0001
n_layers = 2
dropout = 0.3
teacher_forcing_ratio = 0.5
clip_threshold = 1

train_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=True, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size=num_seq*batch_size, shuffle=True)

encoder = Encoder(input_size, hidden_size, n_layers, dropout)
decoder = Decoder(hidden_size, emb_size, output_size, n_layers, dropout)
embed = Embed(output_size, emb_size, dropout)
CNN = CNNaccum(output_size, emb_size, hidden_size, num_seq, exp_size=exp_size, vertical=vertical)

encoder.cuda()
decoder.cuda()
embed.cuda()
CNN.cuda()

encoder.train()
decoder.train()
embed.train()
CNN.train()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
embed_optimizer = optim.Adam(embed.parameters(), lr=learning_rate)
CNN_optimizer =  optim.Adam(CNN.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # 60000 datas
    start_time = time.time()
    k=0
    total_acc = 0
    total_loss = 0
    for data, target in train_loader:
        # num_seq = randint(1,MAXLENGTH)
        loss = 0
        batch_xs = torch.zeros([batch_size, 28*num_seq, 28])
        batch_ys = torch.zeros([batch_size, num_seq+2, 1])
        batch_ys[:,0,:] = 10 #<sos>
        batch_ys[:,-1,:] = 11 #<eos>

        for batch_iterator in range(batch_size):
            x = torch.cat([data[j] for j in range(batch_iterator*num_seq, (batch_iterator+1)*num_seq)], 2)
            x = x.transpose(1,2)
            batch_xs[batch_iterator] = x
            y = target[batch_iterator*num_seq:(batch_iterator+1)*num_seq].view(-1,1)
            batch_ys[batch_iterator][1:num_seq+1] = y
        
        batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device) # Set variables

        embed_optimizer.zero_grad()
        encoder_optimizer.zero_grad()
        decoder_optimizer.zero_grad()
        CNN_optimizer.zero_grad()
        criterion = torch.nn.CrossEntropyLoss()

        decoder_output = torch.zeros(num_seq+1, batch_size, output_size).to(device)
        input = batch_ys[:,0,:].long()
        embedded = embed(input)
        for i in range(num_seq+1):

            # concat aux_context with encoder input
            aux_context = CNN(embedded.long())
            current_batch_xs = torch.cat((batch_xs, aux_context), 1 if vertical else 2)

            hidden, cell = encoder.initHidden(batch_size)
            hidden, cell = encoder(current_batch_xs, hidden, cell)

            # predict current token by calculating previous tokens passed through decoder
            for j in range(i+1):
                output, hidden, cell = decoder(embedded, hidden, cell)
                # teacher_force = random.random() < teacher_forcing_ratio
                input = batch_ys[:,i+1,:].long() # if (teacher_force and i < num_seq) else output.argmax(2)
                embedded = embed(input)
            decoder_output[i] = output.squeeze(1)
            loss += criterion(output.squeeze(1), batch_ys[:,i+1].long().squeeze(1))

        # loss = criterion(decoder_output, target)
        loss.backward() # back propagation
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_threshold) # clipping
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_threshold) # clipping
        torch.nn.utils.clip_grad_norm_(embed.parameters(), clip_threshold) # clipping
        torch.nn.utils.clip_grad_norm_(CNN.parameters(), clip_threshold) # clipping

        embed_optimizer.step() # forward propagation
        CNN_optimizer.step() # forward propagation
        encoder_optimizer.step() # forward propagation
        decoder_optimizer.step() # forward propagation
        
        ### accuracy calculation
        prediction = torch.zeros(batch_size, num_seq).to(device)
        decoder_output = decoder_output[:-1].transpose(0,1) ### without <eos>
        # decoder_output = decoder_output.transpose(0,1)

        prediction = decoder_output.argmax(2).unsqueeze(2)
        accuracy = prediction.eq(batch_ys[:,1:-1,:]).sum()*100/(num_seq*batch_size) ### without <eos>
        # accuracy = prediction.eq(batch_ys[:,1:,:]).sum()*100/(num_seq*batch_size) # eq = count equal
        total_acc += accuracy
        avg_acc = total_acc/(k+1)
        total_loss += loss.item()
        avg_loss = total_loss/(k+1)
        k+=1

    end_time = time.time()
    # print(prediction[-1].transpose(0,1), batch_ys[-1,1:-1].transpose(0,1), num_seq)
    print('Epoch: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch, avg_loss, avg_acc))
    print('Elapsed Time: {}s'.format(int(end_time-start_time)))
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
        self.hidden_size = hidden_size
        self.embedded = nn.Embedding(output_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        input = self.dropout(self.embedded(input))
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        output = self.out(output)
        return output, hidden, cell

batch_size=1000
MAXLENGTH=10
epochs=200
num_seq = 5
input_size = 28
output_size = 12
emb_size = 56
hidden_size = 128
learning_rate = 0.01
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
encoder.cuda()
decoder.cuda()
encoder.train()
decoder.train()

encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)

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
            x = x.reshape(1, 28*num_seq, 28)
            # x = torch.cat([data[j].reshape(1,28*28) for j in range(num_seq)], 0)
            # x = x.transpose(0,0).reshape(num_seq, 1, 28*28)
            batch_xs[batch_iterator] = x
            y = target[batch_iterator*num_seq:(batch_iterator+1)*num_seq].view(-1,1)
            batch_ys[batch_iterator][1:num_seq+1] = y
        
        # fig = plt.figure()
        # subplot = fig.add_subplot(1,1,1)
        # subplot.imshow(x, cmap=plt.cm.gray_r)
        # plt.show()
        
        batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device) # Set variables
        
        encoder_optimizer.zero_grad() # set optimizer
        decoder_optimizer.zero_grad() # set optimizer
        criterion = torch.nn.CrossEntropyLoss()

        hidden, cell = encoder.initHidden(batch_size)
        hidden, cell = encoder(batch_xs, hidden, cell)
        # hidden, cell = encoder.initHidden(batch_size)
        # for i in range(num_seq*28):
        #     hidden, cell = encoder(batch_xs[:,i,:].unsqueeze(1), hidden, cell)

        decoder_output = torch.zeros(num_seq+1, batch_size, output_size).to(device)
        input = batch_ys[:,0,:].long()
        for i in range(num_seq+1):
            output, hidden, cell = decoder(input, hidden, cell)
            teacher_force = random.random() < teacher_forcing_ratio
            input = batch_ys[:,i+1,:].long() if (teacher_force and i < num_seq) else output.argmax(2)
            decoder_output[i] = output.squeeze(1)
            loss += criterion(output.squeeze(1), batch_ys[:,i+1].long().squeeze(1))

        # decoder_output = decoder_output.view(-1, output_size)
        # target = target.squeeze(1)

        # loss = criterion(decoder_output, target)
        loss.backward() # back propagation
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_threshold) # clipping
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_threshold) # clipping

        encoder_optimizer.step() # forward propagation
        decoder_optimizer.step() # forward propagation
        
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
    print(prediction[-1].transpose(0,1), batch_ys[-1,1:-1].transpose(0,1), num_seq)
    print('Epoch: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format(epoch, avg_loss, avg_acc))
    print('Elapsed Time: {}s'.format(int(end_time-start_time)))

## Testing ##
encoder.eval()
decoder.eval()

batch_size = 10
test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size=num_seq*batch_size, shuffle=True)

total_acc = 0
k = 0
for data, target in test_loader:

    batch_xs = torch.zeros([batch_size, 28*num_seq, 28])
    batch_ys = torch.zeros([batch_size, num_seq+2, 1])
    for batch_iterator in range(batch_size):
        x = torch.cat([data[j] for j in range(batch_iterator*num_seq, (batch_iterator+1)*num_seq)], 2)
        x = x.reshape(1, 28*num_seq, 28)
        batch_xs[batch_iterator] = x
        y = target[batch_iterator*num_seq:(batch_iterator+1)*num_seq].view(-1,1)
        batch_ys[batch_iterator][1:num_seq+1] = y
    batch_ys[:,0,:] = 10 #<sos>
    batch_ys[:,-1,:] = 11 #<eos>

    batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)
    
    hidden, cell = encoder.initHidden(batch_size)
    hidden, cell = encoder(batch_xs, hidden, cell)
    # for i in range(num_seq*28):
    #     hidden, cell = encoder(batch_xs[:,i,:].unsqueeze(1), hidden, cell)

    decoder_output = torch.zeros(num_seq+1, batch_size, output_size).to(device)
    input = batch_ys[:,0,:].long()
    for i in range(num_seq):
        output, hidden, cell = decoder(input, hidden, cell)
        input =  output.argmax(2)
        decoder_output[i] = output.squeeze(1)

    prediction = torch.zeros(batch_size, num_seq).to(device)
    decoder_output = decoder_output[:-1].transpose(0,1)
    prediction = decoder_output.argmax(2).unsqueeze(2)
    accuracy = prediction.eq(batch_ys[:,1:-1]).sum()*100/(num_seq*batch_size) # eq = count equal
    total_acc += accuracy
    avg_acc = total_acc/(k+1)
    k+=1

print('Test Accuracy: {:.3f}'.format(avg_acc))
print(prediction[-1].transpose(0,1), batch_ys[-1,1:-1].transpose(0,1), num_seq)
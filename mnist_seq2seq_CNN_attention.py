### Two seq2seq models: attention, CNN accumulated
### CNN accumulation model input = attention model output

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
        self.dropout = nn.Dropout(dropout)
        self.embedded = nn.Embedding(output_size, emb_size)
        self.lstm = nn.LSTM(emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        embedded = self.dropout(self.embedded(input))
        output, (hidden, cell) = self.lstm(embedded, (hidden, cell))
        output = self.out(output)
        return output, hidden, cell

class AttEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout):
        super(AttEncoder, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        # self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell

    def initHidden(self, batch_size):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return hidden, cell

class AttDecoder(nn.Module):
    def __init__(self, hidden_size, emb_size, output_size, n_layers, dropout):
        super(AttDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.embedded = nn.Embedding(output_size, emb_size)
        self.hid_lin = nn.Linear(hidden_size, hidden_size)
        self.enc_lin = nn.Linear(hidden_size, hidden_size)
        self.tan_weight = nn.Parameter(torch.FloatTensor(batch_size, hidden_size, 1).to(device))
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size*n_layers + emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedded(input)) # embedded.size() = [batch_size, 1, emb_size]
        context_vec = torch.zeros(self.n_layers, batch_size, 1, self.hidden_size, device=device)
        attention = torch.zeros(num_seq*28).to(device)

        for layer_idx in range(self.n_layers):
            # hidden.size() = [n_layers, batch_size, hidden_size]            
            tan = torch.tanh(self.hid_lin(hidden[layer_idx].squeeze(0))+self.enc_lin(encoder_outputs)) # tan.size() = [num_seq*28, batch_size, hidden_size]
            tan = tan.transpose(0,1) # tan.size() = [batch_size, num_seq*28, hidden_size]
            att_weight = F.softmax(tan.bmm(self.tan_weight).squeeze(2), dim=1) # att_weight.size() = [batch_size, num_seq*28]
            attention += att_weight[-1]
            e_out_transposed = encoder_outputs.transpose(0,1) # e_out_transposed.size() = [batch_size, num_seq*28, hidden_size]
            context_vec[layer_idx] = torch.bmm(att_weight.unsqueeze(1), e_out_transposed)
            
        context_vec_cat = torch.cat(tuple(context_vec[i] for i in range(self.n_layers)), 2) # context_vec_cat.size() = [batch_size, 1, hidden_size*n_layers]
        new_input = torch.cat((embedded, context_vec_cat), 2) # attention applied input size = [batch_size, 1, hidden_size*n_layers + emb_size]

        new_input = F.relu(new_input)
        output, (hidden, cell) = self.lstm(new_input, (hidden, cell))
        output = self.out(output)

        return output, hidden, cell, (attention/n_layers)

class CNNaccum(nn.Module):
    def __init__(self, output_size, emb_size, hidden_size, num_seq, kernel_size=3, n_layers=3, accum_param=1):
        super(CNNaccum, self).__init__()
        self.scale = torch.sqrt(torch.FloatTensor([0.5])).to(device)
        self.kernel_size = kernel_size
        self.embedding = nn.Embedding(output_size, emb_size)
        self.emb2hid = nn.Linear(emb_size, hidden_size)
        self.fc_out = nn.Linear(hidden_size, 28*num_seq*accum_param)
        self.convs = nn.ModuleList([nn.Conv1d(in_channels = hidden_size, 
                                              out_channels = 2 * hidden_size, 
                                              kernel_size = kernel_size)
                                    for _ in range(n_layers)])
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, input):
        
        batch_size = input.shape[0]
        embedded = self.dropout(self.embedding(input))
        conv_input = self.emb2hid(embedded)
        conv_input = conv_input.squeeze(2).permute(0, 2, 1) # conv_input = [batch size, hidden_size, num_seq+2]
        
        batch_size = conv_input.shape[0]
        hid_dim = conv_input.shape[1]
        
        for i, conv in enumerate(self.convs):
            conv_input = self.dropout(conv_input)

            #zero padding
            padding = torch.zeros(batch_size, 
                                  hidden_size, 
                                  self.kernel_size - 1).fill_(0).to(device)
            padded_conv_input = torch.cat((padding, conv_input), dim = 2) #padded_conv_input = [batch size, hidden_size, num_seq+2 + kernel size - 1]
        
            #pass through convolutional layer
            conved = conv(padded_conv_input) #conved = [batch size, 2 * hidden_size, num_seq+2]
            conved = F.glu(conved, dim = 1) #conved = [batch size, hidden_size, num_seq+2]

            #apply residual connection
            conved = (conved + conv_input) * self.scale
            conv_input = conved
            
        output = self.fc_out(self.dropout(conved.permute(0, 2, 1)))
        
        output = self.fc_out(self.dropout(conved.permute(0, 2, 1))) #output = [batch size, num_seq+2, output_size]
        return output.permute(0,2,1)

def run(batch_xs, batch_ys, mode, attencoder, attdecoder, encoder, decoder, CNN,
 attencoder_optimizer, attdecoder_optimizer, encoder_optimizer, decoder_optimizer, CNN_optimizer, loss=0, teacher_forcing_ratio = 0.5,
 clip_threshold = 1):
    if mode=='ATT':

        attencoder_optimizer.zero_grad() # set optimizer
        attdecoder_optimizer.zero_grad() # set optimizer

        criterion = torch.nn.CrossEntropyLoss()

        encoder_outputs = torch.zeros(num_seq*28, batch_size, hidden_size, device=device)
        hidden, cell = attencoder.initHidden(batch_size)
        for i in range(num_seq*28):
            output, hidden, cell = attencoder(batch_xs[:,i,:].unsqueeze(1), hidden, cell)
            encoder_outputs[i] = output.squeeze(1) # output.size() = [batch_size, 1, hidden_size]

        decoder_output = torch.zeros(num_seq+1, batch_size, output_size).to(device)
        input = batch_ys[:,0,:].long()
        # attention_graph = torch.zeros(num_seq+1, num_seq*28).to(device) # save attention for graph
        for i in range(num_seq+1):
            output, hidden, cell, attention = attdecoder(input, hidden, cell, encoder_outputs)
            # attention_graph[i] = attention
            teacher_force = random.random() < teacher_forcing_ratio
            input = batch_ys[:,i+1,:].long() if (teacher_force and i < num_seq) else output.argmax(2)
            decoder_output[i] = output.squeeze(1)
            loss += criterion(output.squeeze(1), batch_ys[:,i+1].long().squeeze(1))

        loss.backward() # back propagation
        torch.nn.utils.clip_grad_norm_(attencoder.parameters(), clip_threshold) # clipping
        torch.nn.utils.clip_grad_norm_(attdecoder.parameters(), clip_threshold) # clipping

        attencoder_optimizer.step() # forward propagation
        attdecoder_optimizer.step() # forward propagation
        
        decoder_output = decoder_output[:-1].transpose(0,1) ### without <eos>

        return loss, decoder_output

    elif mode=='CNN':
        
        aux_context = CNN(batch_ys.long())
        batch_xs = torch.cat((batch_xs, aux_context), 2)

        encoder_optimizer.zero_grad() # set optimizer
        decoder_optimizer.zero_grad() # set optimizer
        CNN_optimizer.zero_grad()

        criterion = torch.nn.CrossEntropyLoss()

        hidden, cell = encoder.initHidden(batch_size)
        hidden, cell = encoder(batch_xs, hidden, cell)

        decoder_output = torch.zeros(num_seq+1, batch_size, output_size).to(device)
        input = batch_ys[:,0,:].long()
        for i in range(num_seq+1):
            output, hidden, cell = decoder(input, hidden, cell)
            teacher_force = random.random() < teacher_forcing_ratio
            input = batch_ys[:,i+1,:].long() if (teacher_force and i < num_seq) else output.argmax(2)
            decoder_output[i] = output.squeeze(1)
            loss += criterion(output.squeeze(1), batch_ys[:,i+1].long().squeeze(1))


        loss.backward() # back propagation
        torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_threshold) # clipping
        torch.nn.utils.clip_grad_norm_(decoder.parameters(), clip_threshold) # clipping

        CNN_optimizer.step()
        encoder_optimizer.step() # forward propagation
        decoder_optimizer.step() # forward propagation
            
        decoder_output = decoder_output[:-1].transpose(0,1)

        return loss, decoder_output

    elif mode=='TEST':
        ### ATT
        encoder_outputs = torch.zeros(num_seq*28, batch_size, hidden_size, device=device)
        hidden, cell = attencoder.initHidden(batch_size)
        for i in range(num_seq*28):
            output, hidden, cell = attencoder(batch_xs[:,i,:].unsqueeze(1), hidden, cell)
            encoder_outputs[i] = output.squeeze(1) # output.size() = [batch_size, 1, hidden_size]

        decoder_output = torch.zeros(num_seq+1, batch_size, output_size).to(device)
        input = batch_ys[:,0,:].long()
        # attention_graph = torch.zeros(num_seq+1, num_seq*28).to(device) # save attention for graph
        for i in range(num_seq+1):
            output, hidden, cell, attention = attdecoder(input, hidden, cell, encoder_outputs)
            # attention_graph[i] = attention
            input = output.argmax(2)
            decoder_output[i] = output.squeeze(1)

        decoder_output = decoder_output[:-1].transpose(0,1)
        attdecoder_output = decoder_output

        ### CNN
        decoded_out = torch.zeros(batch_size, num_seq+2, 1).to(device)
        decoded_out[:,0,:] = 10 #<sos>
        decoded_out[:,-1,:] = 11 #<eos>
        decoded_out[:,1:-1,:] = decoder_output.argmax(2).unsqueeze(2)
        aux_context = CNN(decoded_out.long())
        batch_xs = torch.cat((batch_xs, aux_context), 2)

        hidden, cell = encoder.initHidden(batch_size)
        hidden, cell = encoder(batch_xs, hidden, cell)

        decoder_output = torch.zeros(num_seq+1, batch_size, output_size).to(device)
        input = batch_ys[:,0,:].long()
        for i in range(num_seq+1):
            output, hidden, cell = decoder(input, hidden, cell)
            input = output.argmax(2)
            decoder_output[i] = output.squeeze(1)

        decoder_output = decoder_output[:-1].transpose(0,1)
        cnndecoder_output = decoder_output

        return attdecoder_output, cnndecoder_output

global num_seq
num_seq = 8
accum_param=1
batch_size=125
MAXLENGTH=10
epochs=20
attinput_size = 28
input_size = 38
output_size = 12
emb_size = 128
hidden_size = 256
learning_rate = 0.001
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

attencoder = AttEncoder(attinput_size, hidden_size, n_layers, dropout)
attdecoder = AttDecoder(hidden_size, emb_size, output_size, n_layers, dropout)
encoder = Encoder(input_size, hidden_size, n_layers, dropout)
decoder = Decoder(hidden_size, emb_size, output_size, n_layers, dropout)
CNN = CNNaccum(output_size, emb_size, hidden_size, num_seq)

attencoder.cuda()
attdecoder.cuda()
encoder.cuda()
decoder.cuda()
CNN.cuda()

attencoder.train()
attdecoder.train()
encoder.train()
decoder.train()
CNN.train()

attencoder_optimizer = optim.Adam(attencoder.parameters(), lr=learning_rate)
attdecoder_optimizer = optim.Adam(attdecoder.parameters(), lr=learning_rate)
encoder_optimizer = optim.Adam(encoder.parameters(), lr=learning_rate)
decoder_optimizer = optim.Adam(decoder.parameters(), lr=learning_rate)
CNN_optimizer =  optim.Adam(CNN.parameters(), lr=learning_rate)

for epoch in range(epochs):
    # 60000 datas
    start_time = time.time()
    k=0
    att_total_acc = 0
    att_total_loss = 0
    total_acc = 0
    total_loss = 0
    for data, target in train_loader:

        model_args = (attencoder, attdecoder, encoder, decoder, CNN)
        optim_args = (attencoder_optimizer, attdecoder_optimizer, encoder_optimizer, decoder_optimizer, CNN_optimizer)
        
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
        
        mode = 'ATT'
        loss, decoder_output = run(batch_xs, batch_ys, mode, *model_args, *optim_args)
        
        prediction = decoder_output.argmax(2).unsqueeze(2)
        accuracy = prediction.eq(batch_ys[:,1:-1,:]).sum()*100/(num_seq*batch_size) ### without <eos>
            
        att_total_acc += accuracy
        att_avg_acc = att_total_acc/(k+1)
        att_total_loss += loss.item()
        att_avg_loss = att_total_loss/(k+1)
        
        force_threshold = 0
        teacher_force = random.random() > 0 #teacher_forcing_ratio
        if (att_avg_acc > force_threshold) and teacher_force:
            batch_ys[:,1:-1,:] = prediction

        mode = 'CNN'
        loss, decoder_output = run(batch_xs, batch_ys, mode, *model_args, *optim_args)
        prediction = decoder_output.argmax(2).unsqueeze(2)
        accuracy = prediction.eq(batch_ys[:,1:-1,:]).sum()*100/(num_seq*batch_size) ### without <eos>
        
        total_acc += accuracy
        avg_acc = total_acc/(k+1)
        total_loss += loss.item()
        avg_loss = total_loss/(k+1)
        k+=1

    print('Mode: {}\tEpoch: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format('ATT', epoch, att_avg_loss, att_avg_acc))
    print('Mode: {}\tEpoch: {}\tLoss: {:.3f}\tAccuracy: {:.3f}'.format('CNN', epoch, avg_loss, avg_acc))

    end_time = time.time()
    print('Elapsed Time: {}s'.format(int(end_time-start_time)))    

# save model
model_dir = './model'
torch.save(attencoder.state_dict(), model_dir+'/attencoder_v8'+'_'+str(learning_rate))
torch.save(attdecoder.state_dict(), model_dir+'/attdecoder_v8'+'_'+str(learning_rate))
torch.save(encoder.state_dict(), model_dir+'/encoder_v8'+'_'+str(learning_rate))
torch.save(decoder.state_dict(), model_dir+'/decoder_v8'+'_'+str(learning_rate))
torch.save(CNN.state_dict(), model_dir+'/CNN_v8'+'_'+str(learning_rate))

## Testing ##

# load model
model_dir = './model'
attencoder = AttEncoder(attinput_size, hidden_size, n_layers, dropout)
attdecoder = AttDecoder(hidden_size, emb_size, output_size, n_layers, dropout)
encoder = Encoder(input_size, hidden_size, n_layers, dropout)
decoder = Decoder(hidden_size, emb_size, output_size, n_layers, dropout)
CNN = CNNaccum(output_size, emb_size, hidden_size, num_seq)

attencoder.load_state_dict(torch.load(model_dir+'/attencoder_v8'+'_'+str(learning_rate)))
attdecoder.load_state_dict(torch.load(model_dir+'/attdecoder_v8'+'_'+str(learning_rate)))
encoder.load_state_dict(torch.load(model_dir+'/encoder_v8'+'_'+str(learning_rate)))
decoder.load_state_dict(torch.load(model_dir+'/decoder_v8'+'_'+str(learning_rate)))
CNN.load_state_dict(torch.load(model_dir+'/CNN_v8'+'_'+str(learning_rate)))

attencoder.cuda()
attdecoder.cuda()
encoder.cuda()
decoder.cuda()
CNN.cuda()

attencoder.eval()
attdecoder.eval()
encoder.eval()
decoder.eval()
CNN.eval()

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST('./data', train=False, download=False,
        transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
            ])),
    batch_size=num_seq*batch_size, shuffle=True)

mode = 'TEST'
atttotal_acc = 0
cnntotal_acc = 0
k = 0

model_args = (attencoder, attdecoder, encoder, decoder, CNN)
optim_args = (0,0,0,0,0)

for data, target in test_loader:

    batch_xs = torch.zeros([batch_size, 28*num_seq, 28])
    batch_ys = torch.zeros([batch_size, num_seq+2, 1])
    for batch_iterator in range(batch_size):
        x = torch.cat([data[j] for j in range(batch_iterator*num_seq, (batch_iterator+1)*num_seq)], 2)
        x = x.transpose(1,2)
        batch_xs[batch_iterator] = x
        y = target[batch_iterator*num_seq:(batch_iterator+1)*num_seq].view(-1,1)
        batch_ys[batch_iterator][1:num_seq+1] = y
    batch_ys[:,0,:] = 10 #<sos>
    batch_ys[:,-1,:] = 11 #<eos>

    batch_xs, batch_ys = batch_xs.to(device), batch_ys.to(device)

    attdecoder_output, cnndecoder_output = run(batch_xs, batch_ys, mode, *model_args, *optim_args)

    prediction = torch.zeros(batch_size, num_seq).to(device)
    attprediction = attdecoder_output.argmax(2).unsqueeze(2)
    cnnprediction = cnndecoder_output.argmax(2).unsqueeze(2)
    
    attaccuracy = attprediction.eq(batch_ys[:,1:-1]).sum()*100/(num_seq*batch_size) # eq = count equal
    cnnaccuracy = cnnprediction.eq(batch_ys[:,1:-1]).sum()*100/(num_seq*batch_size) # eq = count equal
    atttotal_acc += attaccuracy
    cnntotal_acc += cnnaccuracy
    attavg_acc = atttotal_acc/(k+1)
    cnnavg_acc = cnntotal_acc/(k+1)
    k+=1

print('Mode: {}\tTest Accuracy: {:.3f}'.format('ATT', attavg_acc))
print('Mode: {}\tTest Accuracy: {:.3f}'.format('CNN', cnnavg_acc))
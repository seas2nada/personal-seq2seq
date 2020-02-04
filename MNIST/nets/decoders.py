import torch
from torch import nn
import torch.nn.functional as F

# Basic LSTM Decoder
class Decoder(nn.Module):
    def __init__(self, hidden_size, emb_size, output_size, n_layers, dropout, device):
        super(Decoder, self).__init__()
        self.device = device
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

# Attention-based Decoder
class AttentionDecoder(nn.Module):
    def __init__(self, batch_size, hidden_size, emb_size, output_size, n_layers, dropout, device):
        super(AttentionDecoder, self).__init__()
        self.device = device
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.emb_size = emb_size
        self.output_size = output_size
        self.n_layers = n_layers

        self.embedded = nn.Embedding(output_size, emb_size)
        self.hid_lin = nn.Linear(hidden_size, hidden_size)
        self.enc_lin = nn.Linear(hidden_size, hidden_size)
        self.tan_weight = nn.Parameter(torch.FloatTensor(self.batch_size, hidden_size, 1).to(self.device))
        self.dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(hidden_size + emb_size, hidden_size, n_layers, batch_first=True, dropout=dropout)
        self.out = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden, cell, encoder_outputs):
        embedded = self.dropout(self.embedded(input)) # embedded.size() = [batch_size, 1, emb_size]

        # hidden.size() = [n_layers, batch_size, hidden_size]
        tan = torch.tanh(self.hid_lin(hidden[self.n_layers-1].squeeze(0))+self.enc_lin(encoder_outputs)) # tan.size() = [max_length*28, batch_size, hidden_size]
        tan = tan.transpose(0,1) # tan.size() = [batch_size, max_length*28, hidden_size]
        att_weight = F.softmax(tan.bmm(self.tan_weight).squeeze(2), dim=1) # att_weight.size() = [batch_size, max_length*28]

        # save attention for graph
        attention = att_weight[-1]

        e_out_transposed = encoder_outputs.transpose(0,1) # e_out_transposed.size() = [batch_size, max_length*28, hidden_size]
        context_vec = torch.bmm(att_weight.unsqueeze(1), e_out_transposed) # context_vec.size() = [batch_size, 1, hidden_size]

        new_input = torch.cat((embedded, context_vec), 2) # attention applied input size = [batch_size, 1, hidden_size*n_layers + emb_size]

        new_input = F.relu(new_input)
        output, (hidden, cell) = self.lstm(new_input, (hidden, cell)) # pass through LSTM with new input
        output = self.out(output)

        return output, hidden, cell, attention

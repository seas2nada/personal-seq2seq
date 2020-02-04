import torch
from torch import nn
import torch.nn.functional as F

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, n_layers, dropout, device):
        super(RNN, self).__init__()
        self.device = device
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.n_layers = n_layers

        self.lstm = nn.LSTM(input_size, hidden_size, n_layers, batch_first=True, dropout=dropout)

    # basic lstm encoder
    def forward(self, input, hidden, cell):
        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell

    # initiate hidden
    def initHidden(self, batch_size, device):
        hidden = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        cell = torch.zeros(self.n_layers, batch_size, self.hidden_size, device=device)
        return hidden, cell

# torch modules
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from torchvision import datasets, transforms

# basic libraries
import os
import time
import random
from random import randint

# outter libraries
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# internal modules
from utils.directories import CheckDir, FileExists, ModelDir
from utils.plot import PlotAttention
from nets.run_seq2seq import train, test

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser(parser=None, required=True):
    if parser is None:
        parser = argparse.ArgumentParser()

    # data related arguments
    parser.add_argument('--batch_size', type=int, required=required, help='data batch size')
    parser.add_argument('--max_length', type=int, required=required, help='max sequence length')

    # encoder arguments
    parser.add_argument('--input_size', type=int, required=required, help='encoder input size')
    parser.add_argument('--hidden_size', type=int, required=required, help='encoder hidden size')
    parser.add_argument('--n_layers', type=int, required=required, help='encoder number of layers')
    parser.add_argument('--dropout', type=float, required=required, help='dropout rate')

    # decoder arguments
    parser.add_argument('--emb_size', type=int, required=required, help='decoder embedding size')
    parser.add_argument('--output_size', type=int, required=required, help='decoder output size')

    # train arguments
    parser.add_argument('--epochs', default=20, type=int, required=required, help='number of epochs')
    parser.add_argument('--learning_rate', default=0.001, type=float, required=required, help='learning rate')
    parser.add_argument('--teacher_forcing_ratio', default=0.5, type=float, required=required, help='teacher forcing ratio')
    parser.add_argument('--clip_threshold', default=1, type=float, required=required, help='gradient clip threshold')

    return parser

# return train/test data
def get_data(data_dir, get_number, train=False):
    loader = torch.utils.data.DataLoader(
        datasets.MNIST(data_dir, train=train, download=True,
            transform=transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))
                ])),
        batch_size=get_number, shuffle=True)

    return loader

def main():
    # decode: decide whether to print output results or not while training/testing
    decode = True

    # get parsed datas
    parser = get_parser()
    args = parser.parse_args()

    # save directory setting
    data_dir = './data'
    graph_dir = './graph2'
    model_save_dir = './model2'
    dir_list = [data_dir, graph_dir, model_save_dir]
    CheckDir(dir_list)

    # get data
    get_number = args.max_length*args.batch_size
    train_loader = get_data(data_dir, get_number, train=True)
    test_loader = get_data(data_dir, get_number, train=False)

    # train
    train(train_loader, args, model_save_dir, graph_dir, device, decode = decode)

    # test
    # enc_load_dir, dec_load_dir = ModelDir(model_save_dir, last epoch index, learning_rate)
    enc_load_dir, dec_load_dir = ModelDir(model_save_dir, args.epochs-1, args.learning_rate)
    test(test_loader, args, enc_load_dir, dec_load_dir, device)

if __name__=="__main__":
    main()
    exit()

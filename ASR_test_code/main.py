# torch modules
import torch
from torch import nn
import torch.nn.functional as F
import torch.optim as optim

# basic libraries
import os
import sys
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
from utils.data_generator import GetData
from utils.plot import PlotAttention
from nets.run_seq2seq import train, test

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser(parser=None, required=True):
    if parser is None:
        parser = argparse.ArgumentParser()

    # data related arguments
    parser.add_argument('--batch_size', type=int, required=required, help='data batch size')
    parser.add_argument('--max_in', type=int, required=required, help='max encoder input sequence length')
    parser.add_argument('--max_out', type=int, required=required, help='max decoder output sequence length')

    # encoder arguments
    parser.add_argument('--input_size', type=int, required=required, help='encoder input size')
    parser.add_argument('--hidden_size', type=int, required=required, help='encoder hidden size')
    parser.add_argument('--elayers', type=int, required=required, help='encoder number of layers')
    parser.add_argument('--dlayers', type=int, required=required, help='decoder number of layers')
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

def main(train_set, test_set):
    # decode: decide whether to print output results or not while training/testing
    decode = True

    # get parsed datas
    parser = get_parser()
    args = parser.parse_args()

    # save directory setting
    graph_dir = './exp/graph'
    model_save_dir = './exp/model/adadel'
    dir_list = [graph_dir, model_save_dir]
    CheckDir(dir_list)

    # get data
    train_feat = train_set + '/feats.scp'
    test_feat = test_set + '/feats.scp'
    train_index = train_set + '/index.txt'
    test_index = test_set + '/index.txt'
    train_loader = GetData(train_feat, train_index, args, device)

    # initial train
    train(train_loader, args, model_save_dir, graph_dir, device, decode = decode)

    # continue train
    # enc_load_dir, dec_load_dir = ModelDir(model_save_dir, args.epochs-1, args.learning_rate)
    # train(train_loader, args, model_save_dir, graph_dir, device, enc_load_dir = enc_load_dir, \
    #       dec_load_dir = dec_load_dir, decode = decode, load = True)

    # test
    test_loader = GetData(test_feat, test_index, args, device)
    # enc_load_dir, dec_load_dir = ModelDir(model_save_dir, last epoch index, learning_rate)
    enc_load_dir, dec_load_dir = ModelDir(model_save_dir, args.epochs-1, args.learning_rate)
    test(test_loader, args, enc_load_dir, dec_load_dir, device, graph=True, graph_dir=graph_dir)

if __name__=="__main__":
    train_set = "data/train"
    test_set = "data/test"
    main(train_set, test_set)
    exit()

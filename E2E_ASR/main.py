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
from nets.run_seq2seq import run

# set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def get_parser(parser=None, required=True):
    if parser is None:
        parser = argparse.ArgumentParser()

    # data related arguments
    parser.add_argument('--batch_size', default=20, type=int, help='data batch size')
    parser.add_argument('--max_in', type=int, required=required, help='max encoder input sequence length')
    parser.add_argument('--max_out', type=int, required=required, help='max decoder output sequence length')

    # encoder arguments
    parser.add_argument('--etype', default="blstmp", type=str, required=required, help='encoder model type')
    parser.add_argument('--input_size', type=int, required=required, help='encoder input size')
    parser.add_argument('--hidden_size', type=int, required=required, help='encoder hidden size')
    parser.add_argument('--elayers', type=int, required=required, help='encoder number of layers')
    parser.add_argument('--subsample', default=None, type=str, help='encoder subsampling')
    parser.add_argument('--dropout', type=float, required=required, help='dropout rate')

    # decoder arguments
    parser.add_argument('--emb_size', type=int, required=required, help='decoder embedding size')
    parser.add_argument('--output_size', type=int, required=required, help='decoder output size')
    parser.add_argument('--dlayers', type=int, required=required, help='decoder number of layers')
    parser.add_argument('--mtl_alpha', default=0, type=float, help='mtl_alpha: 0 for attention mode, 1 for CTC mode, 0~1 values for hybrid CTC-attention')

    # attention arguments
    parser.add_argument('--atype', default="LocAtt", type=str, help='attention type')
    parser.add_argument('--att_size', type=int, help='attention size')
    parser.add_argument('--aconv_chans', default=10, type=int, help='location attention convolution channels')
    parser.add_argument('--aconv_filts', default=100, type=int, help='location attention convolution filters')

    # CTC arguments
    parser.add_argument('--ctc_type', default="warpctc", type=str, help='CTC type')

    # train arguments
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')
    parser.add_argument('--teacher_forcing_ratio', default=0.5, type=float, help='teacher forcing ratio')
    parser.add_argument('--clip_threshold', default=5, type=float, help='gradient clip threshold')

    # optimization arguments
    parser.add_argument('--optimizer', default='adam', type=str, required=required, help='optimizer')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--lr_decay', default=None, type=float, help='learning rate decay rate')
    parser.add_argument('--eps', default=1e-8, type=float, help='epsilon')
    parser.add_argument('--eps_decay', default=None, type=float, help='epsilon decay rate')

    # decoding check option
    parser.add_argument('--decoding_index', default=0, type=int, help='decide to decode nth output')

    return parser

def main(train_set, test_set):

    decode = True   # decide whether to print output results or not while training/testing
    train = True
    test = True
    early_stop = False # stop training early with loss<1
    graph = True    # attention graph

    # get parsed datas
    parser = get_parser()
    args = parser.parse_args()

    # save directory setting
    graph_dir = './exp/graph/' + args.optimizer
    model_dir = './exp/model/' + args.optimizer
    dir_list = [graph_dir, model_dir]
    CheckDir(dir_list)

    # load directory setting
    """
    If initial training: Usage
    model_load_dir = None
    
    If continue training: Usage
    model_load_dir = ModelDir(model_dir, last trained epoch, learning_rate)
    """
    model_load_dir = None

    # get data
    train_feat = train_set + '/feats.scp'
    test_feat = test_set + '/feats.scp'
    train_index = train_set + '/index.txt'
    test_index = test_set + '/index.txt'
    train_seqlen = train_set + '/seq_len.txt'
    test_seqlen = test_set + '/seq_len.txt'
    train_loader = GetData(train_feat, train_index, train_seqlen, args)
    test_loader = GetData(test_feat, test_index, test_seqlen, args)

    assert not(not train and early_stop), "early stop can only be applied when training"

    # train
    if train:
        epoch = run(train, train_loader, args, model_dir, device, graph=graph, graph_dir=graph_dir,
            model_load_dir=model_load_dir, decode=decode, early_stop=early_stop)

    # test
    if test:
        model_load_dir = ModelDir(model_dir, epoch - 1, args.learning_rate) if early_stop else ModelDir(model_dir,
                                                                                                        args.epochs - 1,
                                                                                                        args.learning_rate)
        _ = run(not test, test_loader, args, model_dir, device, graph=graph, graph_dir=graph_dir,
            model_load_dir=model_load_dir, decode=decode)

if __name__=="__main__":
    train_set = "data/train"
    test_set = "data/test"
    main(train_set, test_set)
    exit()

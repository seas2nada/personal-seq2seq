import torch

# basic libraries
import os
import sys

# outter libraries
import argparse

# internal modules
from utils.directories import CheckDir, FileExists, ModelDir
from utils.data_generator import GetData
from nets.run_seq2seq import recog

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
    parser.add_argument('--mtl_alpha', default=0, type=float,
                        help='mtl_alpha: 0 for attention mode, 1 for CTC mode, 0~1 values for hybrid CTC-attention')

    # attention arguments
    parser.add_argument('--atype', default="LocAtt", type=str, help='attention type')
    parser.add_argument('--att_size', type=int, help='attention size')
    parser.add_argument('--aconv_chans', default=10, type=int, help='location attention convolution channels')
    parser.add_argument('--aconv_filts', default=100, type=int, help='location attention convolution filters')

    # CTC arguments
    parser.add_argument('--ctc_type', default="warpctc", type=str, help='CTC type')

    # arguments for loading model directory
    parser.add_argument('--optimizer', default='none', type=str, help='optimizer')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='learning rate')
    parser.add_argument('--epochs', default=20, type=int, help='number of epochs')

    # decoding related arguments
    parser.add_argument('--beam_size', default=10, type=int, help='beam search size')
    parser.add_argument('--penalty', default=0, type=float, help='what is this?')
    parser.add_argument('--maxlenratio', default=0, type=float, help='feature sequence Vs output sequence max ratio')
    parser.add_argument('--minlenratio', default=0, type=float, help='feature sequence Vs output sequence min ratio')
    parser.add_argument('--ctc_weight', default=0, type=float, help='ctc result weight')
    parser.add_argument('--ctc_window_margin', type=int, default=0,
                        help="""Use CTC window with margin parameter to accelerate
                            CTC/attention decoding especially on GPU. Smaller magin
                            makes decoding faster, but may increase search errors.
                            If margin=0 (default), this function is disabled""")
    parser.add_argument('--nbest', type=int, default=1,
                        help='Output N-best hypotheses')

    return parser

def main(test_set):
    # get parsed datas
    parser = get_parser()
    args = parser.parse_args()

    # save directory setting
    result_dir = 'exp/results'
    result_log = result_dir + '/decoding_' + str(args.optimizer) + '_' + str(args.mtl_alpha) + 'mtl_alpha_' + str(args.beam_size) + 'beam.txt'
    CheckDir(result_dir)

    # load directory setting
    """
    If initial training: Usage
    model_load_dir = None

    If continue training: Usage
    model_load_dir = ModelDir(model_dir, last trained epoch, learning_rate, mtl_alpha)
    """
    model_dir = 'exp/model/' + args.optimizer
    model_load_dir = ModelDir(model_dir, args.epochs - 1, args.learning_rate, args.mtl_alpha)

    # get data
    test_feat = test_set + '/feats.scp'
    test_index = test_set + '/index.txt'
    test_seqlen = test_set + '/seq_len.txt'
    test_loader = GetData(test_feat, test_index, test_seqlen, args)

    # recognize
    recog(test_loader, args, model_load_dir, result_log, device)

if __name__ == "__main__":
    test_set = "data/test"
    main(test_set)
    exit()

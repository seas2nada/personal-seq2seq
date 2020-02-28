"""
Feature Extraction

Default:
n_mfcc=13
n_mel=80
fs=16000
n_fft=512

You can change it manually
"""

# Basic libraries
import sys
import os
from tqdm import tqdm

# outter modules
import numpy as np
import librosa
from scipy.io import wavfile
import kaldiio

# inner modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.directories import CheckDir, FileExists

def load_data(file_path):
    sr, data = wavfile.read(file_path)
    data = data.astype(np.float)
    return data


def save_data(mel, feat_dir, file_path, max_in, pad=False, pad_front=True):
    max_in = int(max_in)
    if pad:
        feat = np.pad(mel, ((0, 0), (max_in - mel.shape[1], 0)), 'constant') if pad_front else np.pad(mel, (
        (0, 0), (0, max_in - mel.shape[1])), 'constant')
    else:
        feat = mel
    npy_file = feat_dir + '/' + file_path.split('/')[-1].split('.wav')[0] + '.npy'
    np.save(npy_file, feat)
    return npy_file

def save_seqlen(npy_file, len_txt):
    feat = np.load(npy_file)
    seq_len = feat.shape[1]

    with open(len_txt, 'a') as f:
        f.write(str(seq_len) + '\n')

def convert_to_mfcc(data, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc)

    mean = np.mean(mfcc)
    std = np.std(mfcc)

    mfcc = (mfcc-mean)/std

    return mfcc

def convert_to_mel(data, fs=16000, window_length_ms=25, shift_length_ms=10, n_mels=80, n_fft=512):
    fmel = librosa.filters.mel(fs, n_fft, n_mels=n_mels)
    wav_stft = np.abs(librosa.core.stft(data, n_fft=n_fft,
                                        hop_length=int(shift_length_ms * fs / 1000),
                                        win_length=int(window_length_ms * fs / 1000),
                                        window='hamming'))
    mel = np.matmul(fmel, wav_stft)
    mel = librosa.power_to_db(mel)
    return mel

if len(sys.argv) != 5:
    print('Usage: python3 extract_feat.py [set] [max_in] [ftype] [input_size]')
    sys.exit(1)

# directory setting
set = sys.argv[1]  # first argument
max_in = sys.argv[2]
ftype = sys.argv[3]
input_size = sys.argv[4]

wavdir = set + '/wav.dir' # load wav file directory
feat_dir = set + '/feats' # feature save directory
len_txt = feat_dir + '/../seq_len.txt'
CheckDir([feat_dir])
if FileExists(len_txt):
    os.remove(len_txt)

with open(wavdir, 'r') as f:
    wavdirs = f.readlines()
    f.close()

# save to mel/mfcc
for file_path in tqdm(wavdirs):
    data = load_data(file_path.strip('\n'))
    if ftype=="mfcc":
        mfcc = convert_to_mfcc(data, n_mfcc=input_size)
        npy_file = save_data(mfcc, feat_dir, file_path.strip('\n'), max_in, pad_front=False)
    elif ftype=="mel":
        mel = convert_to_mel(data, n_mels=input_size)
        npy_file = save_data(mel, feat_dir, file_path.strip('\n'), max_in, pad_front=False)
    save_seqlen(npy_file, len_txt)


# Basic libraries
import sys
import os
from tqdm import tqdm

# outter modules
import numpy as np
import librosa
from scipy.io import wavfile

# inner modules
sys.path.append(os.path.dirname(os.path.abspath(os.path.dirname(__file__))))
from utils.directories import CheckDir


def load_data(file_path):
    sr, data = wavfile.read(file_path)
    data = data.astype(np.float)
    return data


def save_data(mel, feat_dir, file_path, max_in, pad_front=True):
    max_in = int(max_in)
    feat = np.pad(mel, ((0, 0), (max_in - mel.shape[1], 0)), 'constant') if pad_front else np.pad(mel, (
    (0, 0), (0, max_in - mel.shape[1])), 'constant')
    np.save(feat_dir + '/' + file_path.split('/')[-1].split('.wav')[0] + '.npy', feat)
    return

def convert_to_mfcc(data, sr=16000, n_mfcc=40, hop_size=10):
    hop_length = int(hop_size * sr / 1000)
    mfcc = librosa.feature.mfcc(y=data, sr=sr, n_mfcc=n_mfcc, hop_length=hop_length)

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


if len(sys.argv) != 3:
    print('Usage: python data_prep.py [set] [max_in]')
    sys.exit(1)

# directory setting
set = sys.argv[1]  # first argument
max_in = sys.argv[2]

wavdir = set + '/wav.dir' # load wav file directory
feat_dir = set + '/feats' # feature save directory
CheckDir([feat_dir])

with open(wavdir, 'r') as f:
    wavdirs = f.readlines()
    f.close()

# save to mel
for file_path in tqdm(wavdirs):
    data = load_data(file_path.strip('\n'))
    # mfcc = convert_to_mfcc(data, n_mfcc=80)
    # save_data(mfcc, feat_dir, file_path.strip('\n'), max_in, pad_front=False)
    mel = convert_to_mel(data)
    save_data(mel, feat_dir, file_path.strip('\n'), max_in, pad_front=False)

exit()

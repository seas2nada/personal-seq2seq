# End-to-end ASR code
Acoustic Speech Recognition using sequence to sequence model for personal study


# Usage
```shell
bash run.sh
```


# run.sh setting
You can manage nearly whole E2E ASR settings through this file
ex) Feature type, Model type/parameters... etc

## Data setting arguments
### Mini-batch size
Usage: batch_size=20
### Maximum length for padding spectrogram sequence
Usage: max_in=800
### Maximum number of tokens for output sentence
Usage: max_out=60

## Encoder arguments
### Model type for encoder. Choose: lstm, blstm, lstmp, blstmp
Usage: etype="blstmp"
### Feature size (default 80 for mel, 13 for MFCC)
Usage: input_size=80
### Encoder hidden state size
Usage: hidden_size=320
### Encoder number of layers
Usage: elayers=4
### Subsampling rate for lstmp
### ex) hidden_size=320, subsample=1_2_2, then encoder_output size => 80
Usage: subsample="1_2_2_1_1"

## Decoder arguments
### Number of all tokens. Automatically fixed during stage 2
Usage: output_size=29
### Decoder hidden state and embedding vector size
Usage: emb_size=320
### Decoder number of layers
Usage: dlayers=1
### Parameter to choose Multi Task Learning mode
### 0 for attention mode, 1 for CTC mode, 0~1 for CTC-attention hybrid mode
Usage: mtl_alpha=0.3

## Attention arguments
### Attention channel size
Usage: att_size=320
### Attention type. Choose: NoAtt, AddAtt, DotAtt, LocAtt
Usage: atype="LocAtt"
### Number of convolution channels and filters for Location-aware attention
Usage: aconv_chans=10
Usage: aconv_filts=100

## CTC arguments
### Module type for CTC. Choose: warpctc, builtin
Usage: ctc_type="warpctc"


## Training arguments
Usage: epochs=100
Usage: dropout=0.2
Usage: teacher_forcing_ratio=0.5
### Gradient clipping threshold for optimizing gradient descent
Usage: clip_threshold=5

## Optimization arguments
### Optimizer type. Choose: adam, adadelta
Usage: optimizer='adadelta'
### Learning rate for adam
Usage: learning_rate=1
### Epsilon / epsilon decay for adadelta
Usage: eps=1e-8
Usage: eps_decay=0.1

## Combined/Train/Test directory
### Use present data using Kaldi. text, text.txt should be prepared in all dataset directories
Usage: whole_set=data/Combined_data
Usage: train_set=data/train
Usage: test_set=data/test

### Decoding index = n (printout nth output for decoding)
Usage: decoding_index=5


# main.py setting
You can manage train, decoding options through this file

## Arguments
### Decode
Choose to printout result for every epoch
Usage: decode=True
### Train-mode
True for train mode, False for only test
Usage: train=True
### Test-mode
False for not testing
Usage: test=True
### Early stop training
Stop training early when True. By default, training stops when loss is less than 1. You can change it manually through modifying nets/run_seq2seq.py
Usage: early_stop=True
### Save attention graph for nth output, when n=decoding_index
Usage: graph=True
### Directories
You can manually change save/load directories, such as graph_dir, model_dir ... etc, but do not recommend
Recommend to modify only model_load_dir, when load model and continue training or testing

#!/bin/sh

# feature arguments
ftype="mel"
input_size=80 # number of feature

# data setting arguments
batch_size=20
max_in=800 # Dict01 max frame length=1360
max_out=60

# encoder arguments
etype="blstmp"
hidden_size=320
elayers=4
subsample="1_2_2_1_1"

# decoder arguments
output_size=30
emb_size=320
dlayers=1
mtl_alpha=0.3

# attention arguments
att_size=320
atype="LocAtt"
aconv_chans=10
aconv_filts=100

# CTC arguments
ctc_type="warpctc"

# training arguments
epochs=100
dropout=0.2
teacher_forcing_ratio=0.5
clip_threshold=5

# optimization arguments
optimizer='adadelta'
learning_rate=0.001
lr_decay=0.1
eps=1e-8
eps_decay=0.1

# train/test directory
# use present data in Kaldi format
whole_set=data/Combined_data
train_set=data/train
test_set=data/test

# decode index without beam search (n_th output)
decoding_index=5

# decoding related arguments
beam_size=10
penalty=0
maxlenratio=0.0
minlenratio=0.0
ctc_weight=0

# stage
stage=4

if [ ${stage} -le 1 ]; then
  # extract features
  echo 'stage 1: Feature Extraction'
  for set in ${train_set} ${test_set}; do
    if [ -d ${set}/feats ]; then
      rm -rf ${set}/feats
    fi
    python3 local/extract_feat.py ${set} ${max_in} ${ftype} ${input_size}
  done
fi

if [ ${stage} -le 2 ]; then
  # tokenize sentence
  echo 'stage 2: Sentence Tokenizing'

  # token directory
  dict=data/lang
  if [ ! -d ${dict} ]; then
    mkdir ${dict}
  fi
  mkdir temp

  # make dict using whole set
  # google sentencepiece model
  spm_train --input=${whole_set}/text.txt --model_prefix=token_3k --vocab_size=30
  spm_encode --model=token_3k.model --output_format=piece <${whole_set}/text.txt> temp/tkn_text.txt
  cat temp/tkn_text.txt | sed 's/^▁ //g' > temp/tk_text.txt
  echo "<sos>" >> temp/token_word.txt
  echo "<eos>" >> temp/token_word.txt
  echo "<blank>" >> temp/token_word.txt
  cat temp/tk_text.txt | sed 's/ /\n/g' | sed '/^$/d' | LC_COLLATE="ko_KR.UTF-8" sort -u >> temp/token_word.txt

  output_size=`wc -l temp/token_word.txt | awk -F ' ' '{print $1}'`
  echo 'number of tokenized words: '${output_size}

  # save token dictionary
  seq 0 $[output_size-1] > temp/token_index.txt
  paste temp/token_word.txt temp/token_index.txt > ${dict}/tokens.txt

  for set in ${train_set} ${test_set}; do
    # save file name match to utterance
    cat ${set}/wav.dir | awk -F '/' '{print $NF}' | sed 's/.wav//g' > temp/fname
    cat temp/fname | awk -v a=${set}/feats/ -v b=".npy" '{print a$1b}' > ${set}/feats.scp

    # text to index
    spm_encode --model=token_3k.model --output_format=piece <${set}/text.txt> temp/tkn_text.txt
    cat temp/tkn_text.txt | sed 's/^▁ //g' > temp/tk_text.txt
    python3 utils/text2index.py temp/tk_text.txt ${dict}/tokens.txt ${set}/index.txt
    paste temp/tk_text.txt ${set}/index.txt > ${set}/text2index
    paste ${set}/feats.scp ${set}/index.txt > ${set}/feat2index
  done
  rm -rf temp token_3k.model token_3k.vocab
fi

if [ ${stage} -le 3 ]; then
  # prepare training
  echo 'stage 3: Network Training'
  python3 main.py \
        --batch_size ${batch_size} \
        --max_in ${max_in} \
        --max_out ${max_out} \
        --etype ${etype} \
        --atype ${atype} \
        --ctc_type ${ctc_type} \
        --input_size ${input_size} \
        --output_size ${output_size} \
        --emb_size ${emb_size} \
        --hidden_size ${hidden_size} \
        --att_size ${att_size} \
        --aconv_chans ${aconv_chans} \
        --aconv_filts ${aconv_filts} \
        --mtl_alpha ${mtl_alpha} \
        --elayers ${elayers} \
        --dlayers ${dlayers} \
        --subsample ${subsample} \
        --dropout ${dropout} \
        --epochs ${epochs} \
        --learning_rate ${learning_rate} \
        --teacher_forcing_ratio ${teacher_forcing_ratio} \
        --clip_threshold ${clip_threshold} \
        --optimizer ${optimizer} \
        --decoding_index ${decoding_index}
  exit
fi

if [ ${stage} -le 4 ]; then
  # prepare training
  echo 'stage 4: Decoding'
  python3 asr_recog.py \
        --batch_size ${batch_size} \
        --max_in ${max_in} \
        --max_out ${max_out} \
        --etype ${etype} \
        --atype ${atype} \
        --ctc_type ${ctc_type} \
        --input_size ${input_size} \
        --output_size ${output_size} \
        --emb_size ${emb_size} \
        --hidden_size ${hidden_size} \
        --att_size ${att_size} \
        --aconv_chans ${aconv_chans} \
        --aconv_filts ${aconv_filts} \
        --mtl_alpha ${mtl_alpha} \
        --elayers ${elayers} \
        --dlayers ${dlayers} \
        --subsample ${subsample} \
        --dropout ${dropout} \
        --epochs ${epochs} \
        --learning_rate ${learning_rate} \
        --optimizer ${optimizer} \
        --beam_size ${beam_size} \
        --penalty ${penalty} \
        --maxlenratio ${maxlenratio} \
        --minlenratio ${minlenratio} \
        --ctc_weight ${ctc_weight}

fi
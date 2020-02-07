#!/bin/sh

# data setting arguments
batch_size=20
max_in=800 # Dict01 max frame length=1360
max_out=60

# model architecture arguments
input_size=80
output_size=12
emb_size=200
hidden_size=250
elayers=4
dlayers=1
dropout=0.3

# training arguments
epochs=100
learning_rate=1
teacher_forcing_ratio=0.5
clip_threshold=1

# train/test directory
# use present data in Kaldi format
whole_set=data/Combined_data
train_set=data/train
test_set=data/test

# stage
stage=2

if [ ${stage} -le 1 ]; then
  # extract features
  echo 'stage 1: Feature Extraction'
  for set in ${train_set} ${test_set}; do
    if [ -d ${set}/feats ]; then
      rm -rf ${set}/feats
    fi
    python3 local/extract_feat.py ${set} ${max_in}
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
  spm_encode --model=token_3k.model --output_format=piece <${whole_set}/text.txt> temp/tk_text.txt
  echo "<sos>" >> temp/token_word.txt
  echo "<eos>" >> temp/token_word.txt
  cat temp/tk_text.txt | sed 's/ /\n/g' | sed '/^$/d' | LC_COLLATE="ko_KR.UTF-8" sort -u >> temp/token_word.txt

  odim=`wc -l temp/token_word.txt | awk -F ' ' '{print $1}'`
  echo 'number of tokenized words: '${odim}

  # save token dictionary
  seq 0 $[odim-1] > temp/token_index.txt
  paste temp/token_word.txt temp/token_index.txt > ${dict}/tokens.txt

  for set in ${train_set} ${test_set}; do
    # save file name match to utterance
    cat ${set}/wav.dir | awk -F '/' '{print $NF}' | sed 's/.wav//g' > temp/fname
    cat temp/fname | awk -v a=${set}/feats/ -v b=".npy" '{print a$1b}' > ${set}/feats.scp

    # text to index
    spm_encode --model=token_3k.model --output_format=piece <${set}/text.txt> temp/tk_text.txt
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
        --input_size ${input_size} \
        --output_size ${odim} \
        --emb_size ${emb_size} \
        --hidden_size ${hidden_size} \
        --elayers ${elayers} \
        --dlayers ${dlayers} \
        --dropout ${dropout} \
        --epochs ${epochs} \
        --learning_rate ${learning_rate} \
        --teacher_forcing_ratio ${teacher_forcing_ratio} \
        --clip_threshold ${clip_threshold}
fi
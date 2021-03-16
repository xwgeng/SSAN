#!/bin/bash
set -e

export PYTHONPATH=`readlink -f .`:$PYTHONPATH

model=$1
rlsan_enc_list=$2
select_scale=$3
nce_scale=$4

src=en
tgt=de
data_dir=../../data/WMT2014_ende
out_dir=WMT2014_${src}${tgt}/$1
mkdir -p ${out_dir}

python thumt/bin/trainer.py \
    --model $1 \
    --input \
        ${data_dir}/corpus.bpe32k.${src} \
        ${data_dir}/corpus.bpe32k.${tgt} \
    --output ${out_dir} \
    --vocabulary \
        ${data_dir}/vocab.${src}.txt \
        ${data_dir}/vocab.${tgt}.txt \
    --validation \
        ${data_dir}/newstest2013.bpe.${src} \
    --references \
        ${data_dir}/newstest2013.tc.${tgt} \
    --checkpoint \
        WMT2014_${src}${tgt}/ckpt \
    --parameters \
        "batch_size=8192,device_list=[0,1,2,3],train_steps=200000,hidden_size=512,filter_size=2048,num_heads=8,residual_dropout=0.1,attention_dropout=0.1,relu_dropout=0.1,update_cycle=1,rlsan_enc_list=[${rlsan_
    }],select_tokens=True,select_scale=${select_scale},nce_scale=${nce_scale}" \
    > ${out_dir}/log 2>&1 &

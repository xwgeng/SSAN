#!/bin/bash

set -e

export PYTHONPATH=`readlink -f .`:$PYTHONPATH

model=$1 

src=en
tgt=de
data_dir=../../data/WMT2014_ende
out_dir=WMT2014_${src}${tgt}/$1

obj=newstest2014

mkdir -p ${out_dir}/trans

python thumt/bin/translator.py \
    --models $1 \
    --input \
        ${data_dir}/${obj}.bpe.${src} \
    --output \
        ${out_dir}/trans/${obj}.trans \
    --vocabulary \
        ${data_dir}/vocab.${src}.txt \
        ${data_dir}/vocab.${tgt}.txt \
    --checkpoints \
        ${out_dir}/eval \
    --parameters \
        "device_list=[0],decode_batch_size=1"

sed -r 's/(@@ )|(@@ ?$)//g' < ${out_dir}/trans/${obj}.trans > ${out_dir}/trans/${obj}.norm

./multi-bleu.perl ${data_dir}/${obj}.tc.de < $out_dir/trans/${obj}.norm \
        | tee -a ${out_dir}/trans/${obj}.norm.evalResult


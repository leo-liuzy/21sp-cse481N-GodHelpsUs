#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1
model_path=$2
for lang in en zh de es fr ja
do
    ./eval_marc.sh $1 $lang $2
done

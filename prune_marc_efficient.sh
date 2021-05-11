#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1
LANG=$2
model_path=$3
num_layer=12
num_head=16

python run_marc_prune.py --model_name_or_path ${model_path} \
    --output_dir ${model_path} \
    --train_file amazon-reviews-ml/train/dataset_${LANG}_train.json \
    --validation_file amazon-reviews-ml/test/dataset_${LANG}_test.json \
    --label_column_name stars \
    --language $LANG \
    --num_layer $num_layer \
    --num_head $num_head \
    --do_prune \
    --do_eval \
    --per_gpu_train_batch_size 128 \
    --per_gpu_eval_batch_size 128 \
    --gradient_accumulation_steps 1 \
    --num_train_epochs 3 \
    --save_steps -1 \
    --fp16 \
    --max_seq_length 128


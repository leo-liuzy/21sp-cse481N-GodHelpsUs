#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1
LANG=$2
model_path=$3
num_layer=12
num_head=16

for layer in $(seq 0 $((num_layer * 2 - 1)))
do
    if [ $layer -gt $((num_layer - 1)) ]
    then
        # decoder has $num_head more heads
        head_end=$((2 * num_head - 1))
    else
        # encoder has $num_head
        head_end=$((num_head - 1))
    fi
    for head in $(seq 0 $head_end)
    do
        echo "layer: ${layer}, head: ${head}"
        python run_glue.py --model_name_or_path ${model_path} \
            --output_dir ${model_path}/${LANG}/layer${layer}_head${head} \
            --train_file amazon-reviews-ml/train/dataset_${LANG}_train.json \
            --validation_file amazon-reviews-ml/test/dataset_${LANG}_test.json \
            --label_column_name stars \
            --do_prune \
            --layer_idx $layer\
            --head_idx $head \
            --do_eval \
            --per_gpu_train_batch_size 128 \
            --per_gpu_eval_batch_size 128 \
            --gradient_accumulation_steps 1 \
            --num_train_epochs 3 \
            --save_steps -1 \
            --fp16 \
            --max_seq_length 128

    done
done
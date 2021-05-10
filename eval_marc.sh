#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1
LANG=$2
model_path=$3
python run_glue.py --model_name_or_path ${model_path} \
	--output_dir ${model_path} \
	--train_file amazon-reviews-ml/train/dataset_${LANG}_train.json \
       	--validation_file amazon-reviews-ml/test/dataset_${LANG}_test.json \
       	--label_column_name stars \
	--do_eval \
	--per_gpu_train_batch_size 128 \
	--per_gpu_eval_batch_size 128 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 3 \
	--save_steps -1 \
	--fp16 \
	--max_seq_length 128 \

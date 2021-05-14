#!/bin/sh
export CUDA_VISIBLE_DEVICES=$1
LANG=$2
python run_glue.py --model_name_or_path facebook/mbart-large-cc25 \
	--output_dir marc_${LANG}_mbart \
	--train_file amazon-reviews-ml/train/dataset_${LANG}_train.json \
       	--validation_file amazon-reviews-ml/dev/dataset_${LANG}_dev.json \
       	--label_column_name stars \
	--do_train \
	--do_eval \
	--overwrite_output_dir \
	--per_device_train_batch_size 128 \
	--per_device_eval_batch_size 128 \
	--gradient_accumulation_steps 1 \
	--num_train_epochs 3 \
	--save_steps -1 \
	--fp16 \
	--max_seq_length 128 \

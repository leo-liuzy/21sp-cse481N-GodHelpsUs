#!/bin/sh
export CUDA_VISIBLE_DEVICES=0,1
LANG=zh
python run_glue.py --model_name_or_path facebook/mbart-large-cc25 \
	--output_dir marc_${LANG}_mbart \
	--train_file amazon-reviews-ml/train/dataset_${LANG}_train.json \
       	--validation_file amazon-reviews-ml/dev/dataset_${LANG}_dev.json \
       	--label_column_name stars \
	--do_train \
	--overwrite_output_dir \
	--per_gpu_train_batch_size 1 \
	--gradient_accumulation_steps 8 \
	--num_train_epochs 3 \
	--save_steps -1 \
	--fp16 \
	--max_seq_length 32 \

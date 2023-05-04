#!/bin/bash

# Figure 4
# Train a GPT2 model:
python train_GPT2.py --output_dir gpt2_scale --do_train --do_eval --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --save_total_limit 10 --max_eval_samples 1000 --evaluation_strategy steps --max_steps 50000 --config_name gpt2_small_dim_config.json --use_scale_layer_norm

python train_GPT2.py --output_dir gpt2_no_scale --do_train --do_eval --per_device_train_batch_size 16 --per_device_eval_batch_size 16 --save_total_limit 10 --max_eval_samples 1000 --evaluation_strategy steps --max_steps 50000 --config_name gpt2_small_dim_config.json

# Check for unselectable fraction:
# Tables 1, 2
python layer_norm_unselectable_layer.py --model GPT2_W_SCALE --out_dir unselectable_with_scale --min_sample_size 768 --max_sample_size 1024 --dataset SQuAD

python layer_norm_unselectable_layer.py --model GPT2_WO_SCALE --out_dir unselectable_without_scale --min_sample_size 768 --max_sample_size 1024 --dataset SQuAD

# Figure 1c, 1d
python unselectable_figure.py --initialization LAYER_NORM --out_dir unselectable_rand_exp
python unselectable_figure.py --initialization NONE --out_dir unselectable_rand_exp

# Table 3
python layer_norm_unselectable_layer.py --model GPT2_WO_SCALE --out_dir unselectable_without_scale_sst2 --min_sample_size 0 --max_sample_size 128 --dataset SST2

# Table 4
python train_BERT.py --output_dir bert_no_ln --do_train --do_eval --task_name sst2  --per_device_train_batch_size 32 --per_device_eval_batch_size 32 --num_train_epochs 100 --max_seq_length 128 --per_device_eval_batch_size 16 --save_total_limit 10 --max_eval_samples 1000 --config_name bert_small_dim_config.json

python layer_norm_unselectable_layer.py --model BERT_NO_LN --out_dir unselectable_no_ln_bert --min_sample_size 0 --max_sample_size 128 --dataset SST2

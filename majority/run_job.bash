#!/bin/bash
#SBATCH -N 1 # number of minimum nodes
#SBATCH -c 2 # number of cores
#SBATCH --gres=gpu:1 # Request 1 gpu
#SBATCH --job-name="majority"
#SBATCH -o logs/%N.%j.out # stdout goes here
#SBATCH -e logs/%N.%j.out # stderr goes here
#SBATCH --array=11,123,42,9,456,4,5,6,7,8

# random seed
SEED=$SLURM_ARRAY_TASK_ID

python majority_exp.py  --num_samples "${1}"\
                        --seq_len "${2}"\
                        --num_classes "${3}"\
                        --hidden_dim "${4}"\
                        --num_head "${5}"\
                        --epochs "${6}"\
                        --lr "${7}"\
                        --batch_size "${8}"\
                        --data_min_delta "${9}"\
                        --layer_norm "${10}"\
                        --notes "${11}"\
                        --group "${12}"\
                        --mode None\
                        --seed "${SEED}"



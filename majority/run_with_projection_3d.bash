#!/bin/bash

if [[ ! -e logs ]]; then
    mkdir logs
fi

# Start 10 runs with different random seeds

NUM_SAMPLES=6000
SEQ_LEN=30
NUM_CLASSES=10
HIDDEN_DIM=3
NUM_HEAD=1
EPOCHS=10000
LR=0.001
BATCH_SIZE=6000
DATA_MIN_DELTA=6
LAYER_NORM=WITH_PROJ
NOTES="with projection 3d"
# generate group id
GROUP_ID=`python -c 'import wandb; print(wandb.util.generate_id())'`
GROUP="${NOTES}-${GROUP_ID}"
echo "Group ID: $GROUP"
sbatch run_job.bash "${NUM_SAMPLES}" "${SEQ_LEN}" "${NUM_CLASSES}" "${HIDDEN_DIM}" "${NUM_HEAD}" "${EPOCHS}" "${LR}" "${BATCH_SIZE}" "${DATA_MIN_DELTA}" "${LAYER_NORM}" "${NOTES}" "${GROUP}"

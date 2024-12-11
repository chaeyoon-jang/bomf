#!/bin/bash

# Shell script to run HPBO experiments for GLUE tasks
# !! Customizable parameters for large and small tasks !!

EXPERIMENT="python -m experiments.hpbo_glue"

if [ "$1" == "large" ]; then
    TASKS=("mnli" "qqp" "qnli" "sst2")
    BS_LOWER=32
    BS_UPPER=128
    LR_LOWER=1e-6
    LR_UPPER=5e-4
    NUM_GPUS=4
else
    TASKS=("rte" "mrpc")
    BS_LOWER=4
    BS_UPPER=32
    LR_LOWER=1e-6
    LR_UPPER=1e-4
    NUM_GPUS=2
fi

for TASK in "${TASKS[@]}"; do
    echo "Running experiment for task: $TASK"
    $EXPERIMENT \
        --task "$TASK" \
        --lr_lower_bound "$LR_LOWER" \
        --lr_upper_bound "$LR_UPPER" \
        --bs_lower_bound "$BS_LOWER" \
        --bs_upper_bound "$BS_UPPER" \
        --n_epochs 10 \
        --freeze_num 6 \
        --weight_decay 0.1 \
        --warmup_ratio 0.2 \
        --bo_iters 50 \
        --num_workers 4 \
        --dist_url "tcp://127.0.0.1:29500" \
        --world_size "$NUM_GPUS" \
        --rank 0
done

echo "All experiments completed."
#!/bin/bash

# Shell script to run BOMF experiments for SQuAD tasks

if [ $# -lt 8 ]; then
    echo "Usage: $0 <task> <seed> <min_em> <max_em> <min_f1> <max_f1> <n_iters> <base_path>"
    exit 1
fi

TASK=$1
SEED=$2
MIN_EM=$3
MAX_EM=$4
MIN_F1=$5
MAX_F1=$6
N_ITERS=$7
BASE_PATH=$8

EXPERIMENT_SCRIPT="python -m experiments.bomf_squad"

echo "Running experiment for task: $TASK with seed: $SEED"

$EXPERIMENT_SCRIPT \
    --task "$TASK" \
    --seed "$SEED" \
    --min_em "$MIN_EM" \
    --max_em "$MAX_EM" \
    --min_f1 "$MIN_F1" \
    --max_f1 "$MAX_F1" \
    --niters "$N_ITERS" \
    --key "$TASK" \
    --base_path "$BASE_PATH"

echo "Experiment for task: $TASK completed."
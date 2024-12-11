#!/bin/bash

# Shell script to run BOMF experiments for GLUE tasks

if [ $# -lt 8 ]; then
    echo "Usage: $0 <task> <seed> <max_m1> <min_m1> <max_m2> <min_m2> <n_iters> <base_path>"
    exit 1
fi

TASK=$1
SEED=$2
MAX_M1=$3
MIN_M1=$4
MAX_M2=$5
MIN_M2=$6
N_ITERS=$7
BASE_PATH=$8

EXPERIMENT_SCRIPT="python -m experiments.bomf_glue"

echo "Running experiment for task: $TASK"
$EXPERIMENT_SCRIPT \
    --task "$TASK" \
    --seed "$SEED" \
    --max_m1 "$MAX_M1" \
    --min_m1 "$MIN_M1" \
    --max_m2 "$MAX_M2" \
    --min_m2 "$MIN_M2" \
    --n_iters "$N_ITERS" \
    --key "$TASK" \
    --base_path "$BASE_PATH"

echo "Experiment completed."
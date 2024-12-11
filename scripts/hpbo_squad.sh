#!/bin/bash

LR_LOWER_BOUND=1e-7
LR_UPPER_BOUND=1e-3
BS_LOWER_BOUND=8
BS_UPPER_BOUND=16
FREEZE_NUM=0

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --lr_lower_bound) LR_LOWER_BOUND="$2"; shift ;;
        --lr_upper_bound) LR_UPPER_BOUND="$2"; shift ;;
        --bs_lower_bound) BS_LOWER_BOUND="$2"; shift ;;
        --bs_upper_bound) BS_UPPER_BOUND="$2"; shift ;;
        --freeze_num) FREEZE_NUM="$2"; shift ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

BOUNDS_JSON="[[${LR_LOWER_BOUND}, ${BS_LOWER_BOUND}], [${LR_UPPER_BOUND}, ${BS_UPPER_BOUND}]]"

python -m experiments.medium.hpbo_squad \
    --bounds "$BOUNDS_JSON" \
    --freeze_num "$FREEZE_NUM" \
    --bo_seed 42 \
    --train_seed 123 \
    --bo_iters 10 \
    --q 1 \
    --num_restarts 5 \
    --raw_samples 20
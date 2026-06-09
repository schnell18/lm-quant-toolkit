#!/bin/bash

eval_vit_zeroshot_cls() {
    python ../../src/cli.py vit \
        --task eval_zeroshot_cls \
        --model 0 1 2 \
        --algo $ALGO \
        --experiment-name $EXPERIMENT_NAME \
        --quant-snapshot-dir="$SNAPSHOT_DIR" \
        --result-dir="$RESULT_DIR" \
        2>&1 \
        | tee $LOG_DIR/bench-vit-$(date +%Y%m%d%H%M%S).log
}


ALGOS="mxq"
BASE_DIR="/fdata/llm/ieee-tai/vit"
SNAPSHOT_DIR="$BASE_DIR/snapshots"
mkdir -p $SNAPSHOT_DIR

for ALGO in $ALGOS; do
    ATTEMPT="baseline-${ALGO}-2"
    EXPERIMENT_NAME="${ATTEMPT}"
    LOG_DIR="$BASE_DIR/${EXPERIMENT_NAME}/logs"
    RESULT_DIR="$BASE_DIR/${EXPERIMENT_NAME}/results"

    mkdir -p $LOG_DIR
    mkdir -p $RESULT_DIR
    eval_vit_zeroshot_cls
done

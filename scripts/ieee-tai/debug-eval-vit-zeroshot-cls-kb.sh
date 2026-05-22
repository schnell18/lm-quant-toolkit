#!/bin/bash


EXPERIMENT_NAME=zs_kb_trial02
BASE_DIR="/fdata/llm/ieee-tai/vit/$EXPERIMENT_NAME"
LOG_DIR="$BASE_DIR/logs"
RESULT_DIR="$BASE_DIR/results"
SNAPSHOT_DIR="$BASE_DIR/snapshots"

mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $SNAPSHOT_DIR

    # --model 0 1 2 \
python -m pdb ../../src/cli.py vit \
    --task eval_zeroshot_cls \
    --model 0 \
    --algo mxq \
    --weight-algo kurt-boost \
    --boost-stop 1 \
    --top-m-layer 1 \
    --experiment-name $EXPERIMENT_NAME \
    --quant-snapshot-dir="$SNAPSHOT_DIR" \
    --result-dir="$RESULT_DIR" \
    2>&1 \
    | tee $LOG_DIR/bench-vit-$(date +%Y%m%d%H%M%S).log


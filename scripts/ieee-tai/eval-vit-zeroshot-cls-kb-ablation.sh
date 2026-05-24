#!/bin/bash

eval_vit_zeroshot_cls() {
    python ../../src/cli.py vit \
        --task eval_zeroshot_cls \
        --model 0 1 2 \
        --algo mxq \
        --weight-algo kurt-boost \
        --boost-stop $BOOST_STOP \
        --top-m-layer $BOOST_TOP_M \
        --ablation \
        --experiment-name $EXPERIMENT_NAME \
        --quant-snapshot-dir="$SNAPSHOT_DIR" \
        --result-dir="$RESULT_DIR" \
        2>&1 \
        | tee $LOG_DIR/bench-vit-$(date +%Y%m%d%H%M%S).log
}


BOOST_STOPS="1 2"
BOOST_TOP_MS="1 2 3"
BASE_DIR="/fdata/llm/ieee-tai/vit"

SNAPSHOT_DIR="$BASE_DIR/snapshots"
mkdir -p $SNAPSHOT_DIR

for BOOST_STOP in $BOOST_STOPS; do
    for BOOST_TOP_M in $BOOST_TOP_MS; do
        EXPERIMENT_NAME="kurt-boost-ablation-${BOOST_STOP}-${BOOST_TOP_M}"
        LOG_DIR="$BASE_DIR/${EXPERIMENT_NAME}/logs"
        RESULT_DIR="$BASE_DIR/${EXPERIMENT_NAME}/results"

        mkdir -p $LOG_DIR
        mkdir -p $RESULT_DIR
        eval_vit_zeroshot_cls
    done
done

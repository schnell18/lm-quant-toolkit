#!/bin/bash

# BUDGETS="3.13 3.25 3.51 4.13 4.25 4.51"
BUDGETS="4.25"
LOG_DIR="/fdata/llm/ieee-tai/logs"
RESULT_DIR="/fdata/llm/ieee-tai/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/ieee-tai/snapshots"

# Use cached dataset to speedup wikitext, c4 ppl evaluation
export HF_DATASETS_OFFLINE=1
weight_algo=kurt-boost
# MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
# MODEL_NAMES="Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B"
MODELS="Qwen/Qwen3.5-9B"
MODEL_NAMES="Qwen3.5-9B"


# BOOST_STOPS="2 3"
# BOOST_TOP_MS="1 2 3 0"
BOOST_STOPS="2"
BOOST_TOP_MS="1"


for BOOST_STOP in $BOOST_STOPS; do
    for BOOST_TOP_M in $BOOST_TOP_MS; do
        ATTEMPT="kurt-boost-${BOOST_STOP}-${BOOST_TOP_M}"
        EXP_BASE_NAME=$ATTEMPT
        mkdir -p $LOG_DIR
        mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}
        mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT
        mkdir -p $RESULT_DIR/${EXP_NAME}_ppl
        mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,stor}/mxq/$ATTEMPT

        log_file="$LOG_DIR/bench-${ATTEMPT}-$(date +%Y%m%d%H%M%S).log"

        EXP_NAME="${ATTEMPT}"
        echo "=========Quantise Model========="
        python -m pdb ../../src/cli.py llm \
          --task quant \
          --model $MODELS \
          --algo mxq \
          --weight-algo $weight_algo \
          --boost-stop $BOOST_STOP \
          --top-m-layer $BOOST_TOP_M \
          --config ${BUDGETS} \
          --experiment-name "${EXP_NAME}_qnt" \
          --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
          --result-dir=$RESULT_DIR \
          2>&1 \
          | tee -a $log_file
        EXIT_CODE=$?
        if [ $EXIT_CODE -ne 0 ]; then
          echo "quantisation failed!"
          exit $EXIT_CODE
        fi
    done
done

#!/bin/bash

dump_qnt_cfg() {
    echo "=========Dump quantization configs on batch ${EXP_NAME}========="
    mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT
    python -m pdb ../../src/cli.py dump \
      --type quant_config \
      --model $MODELS \
      --budget ${BUDGETS} \
      --attempt $ATTEMPT \
      --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
      --output-file "$RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT/quant-allot-${EXP_NAME}.csv" \
      2>&1 \
      | tee -a $log_file
}

BUDGETS="3.13 3.25 3.51 4.13 4.25 4.51"
LOG_DIR="/fdata/llm/ieee-tai/logs4"
RESULT_DIR="/fdata/llm/ieee-tai/results4"
QUANT_SNAPSHOT_DIR="/fdata/llm/ieee-tai/snapshots4"

# Use cached dataset to speedup wikitext, c4 ppl evaluation
# export HF_DATASETS_OFFLINE=1
weight_algo=kurt-boost
MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
MODEL_NAMES="Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B"
# MODELS="Qwen/Qwen3.5-9B"
# MODEL_NAMES="Qwen3.5-9B"


BOOST_STOPS="2 3"
BOOST_TOP_MS="1 2 3 0"
# BOOST_STOPS="2"
# BOOST_TOP_MS="1"

for BOOST_STOP in $BOOST_STOPS; do
    for BOOST_TOP_M in $BOOST_TOP_MS; do
        ATTEMPT="kurt-boost-${BOOST_STOP}-${BOOST_TOP_M}"
        EXP_BASE_NAME=$ATTEMPT
        EXP_NAME="${ATTEMPT}"

        log_file="$LOG_DIR/bench-${ATTEMPT}-$(date +%Y%m%d%H%M%S).log"
        dump_qnt_cfg
    done
done

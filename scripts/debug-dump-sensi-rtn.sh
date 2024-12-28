#!/bin/bash

QUANT_METHODS="rtn"
# QUANT_METHODS="hqq"
CALIB_DATASETS="pileval"
CONFIGS="b4g64"
MODELS="meta-llama/Llama-2-7b-hf"

RESULT_BASE_DIR="/fdata/llm/mxq/results"
EXP_NAME=debug_hqq_vs_rtn_sensi_llama
RESULT_DIR=$RESULT_BASE_DIR/$EXP_NAME
mkdir -p $RESULT_DIR/data

for QNT_MTD in $QUANT_METHODS; do
    for DS in $CALIB_DATASETS; do
        for CFG in $CONFIGS; do
            for MODEL in $MODELS; do
                SHORT_ID=$(echo $MODEL | cut -d/ -f2)
                OUT_FILE="$RESULT_DIR/${QNT_MTD}-${SHORT_ID}-${CFG}-${DS}.csv"
                python -m pdb ../src/dump.py sensi \
                    --model $MODEL \
                    --quant-method $QNT_MTD \
                    --config $CFG \
                    --calib-dataset $DS \
                    --output-file $OUT_FILE
            done
        done
    done
done

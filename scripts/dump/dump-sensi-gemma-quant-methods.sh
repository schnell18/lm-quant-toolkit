#!/bin/bash

QUANT_METHODS="bnb rtn hqq"
# QUANT_METHODS="bnb"
CALIB_DATASETS="bos pileval wikitext c4"
# CONFIGS="b3g128 b3g64 b3g32 b4g128 b4g64 b4g32 b8g128 b8g64 b8g32"
CONFIGS="b4g128 b4g64 b4g32"
MODELS="google/gemma-7b google/gemma-7b-it google/codegemma-7b google/codegemma-7b-it"

RESULT_BASE_DIR="/fdata/llm/mxq/results"
EXP_NAME=quant_methods_sensi_gemma
RESULT_DIR=$RESULT_BASE_DIR/$EXP_NAME
mkdir -p $RESULT_DIR/data

for QNT_MTD in $QUANT_METHODS; do
    for DS in $CALIB_DATASETS; do
        for CFG in $CONFIGS; do
            for MODEL in $MODELS; do
                SHORT_ID=$(echo $MODEL | cut -d/ -f2)
                OUT_FILE="$RESULT_DIR/${QNT_MTD}-${SHORT_ID}-${CFG}-${DS}.csv"
                python ../src/dump.py sensi \
                    --model $MODEL \
                    --quant-method $QNT_MTD \
                    --config $CFG \
                    --calib-dataset $DS \
                    --output-file $OUT_FILE
            done
        done
    done
done

OLD_DIR=$(pwd)
cd $RESULT_DIR
if [ ! -d pdfs ]; then
    mkdir -p pdfs
fi
$OLD_DIR/../data-vis/plot-quant-method-sensi.R

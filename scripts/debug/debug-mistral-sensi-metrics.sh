#!/bin/bash

RESULT_BASE_DIR="/fdata/llm/mxq/results"
CALIB_DATASETS="pileval"
CONFIGS="b4g64"
MODELS="mistralai/Ministral-8B-Instruct-2410"

EXP_NAME=sensi_mistral
RESULT_DIR=$RESULT_BASE_DIR/$EXP_NAME
mkdir -p $RESULT_DIR/data

for DS in $CALIB_DATASETS; do
    for CFG in $CONFIGS; do
        for MODEL in $MODELS; do
            SHORT_ID=$(echo $MODEL | cut -d/ -f2)
            OUT_FILE="${RESULT_DIR}/data/qwen25-sensi-${SHORT_ID}-${CFG}-${DS}.csv"
            python -m pdb ../src/dump.py sensi \
                --model $MODEL \
                --config $CFG \
                --calib-dataset $DS \
                --output-file $OUT_FILE
        done
    done
done


OLD_DIR=$(pwd)
cd $RESULT_DIR
if [ ! -d pdfs ]; then
    mkdir -p pdfs
fi
$OLD_DIR/../data-vis/plot-variant-sensi.R data/

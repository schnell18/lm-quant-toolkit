#!/bin/bash

RESULT_BASE_DIR="/fdata/llm/mxq/results"
CALIB_DATASETS="bos pileval wikitext c4"
CONFIGS="b2g128 b2g64 b2g32 b3g128 b3g64 b3g32 b4g128 b4g64 b4g32 b8g128 b8g64 b8g32"
MODELS="mistralai/Mistral-7B-v0.3 mistralai/Mistral-7B-Instruct-v0.3 mistralai/Ministral-8B-Instruct-2410"

EXP_NAME=sensi_mistral
RESULT_DIR=$RESULT_BASE_DIR/$EXP_NAME
mkdir -p $RESULT_DIR/data

# for DS in $CALIB_DATASETS; do
#     for CFG in $CONFIGS; do
#         for MODEL in $MODELS; do
#             SHORT_ID=$(echo $MODEL | cut -d/ -f2)
#             OUT_FILE="${RESULT_DIR}/data/mistral-sensi-${SHORT_ID}-${CFG}-${DS}.csv"
#             python ../src/dump.py sensi \
#                 --model $MODEL \
#                 --config $CFG \
#                 --calib-dataset $DS \
#                 --output-file $OUT_FILE
#         done
#     done
# done


OLD_DIR=$(pwd)
cd $RESULT_DIR
if [ ! -d pdfs ]; then
    mkdir -p pdfs
fi
$OLD_DIR/../data-vis/plot-variant-sensi.R data/

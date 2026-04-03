#!/bin/bash

RESULT_BASE_DIR="/fdata/llm/mxq/results"
CALIB_DATASETS="bos pileval wikitext c4"
CONFIGS="b2g128 b2g64 b2g32 b3g128 b3g64 b3g32 b4g128 b4g64 b4g32 b8g128 b8g64 b8g32"
MODELS="Qwen/Qwen2.5-7B Qwen/Qwen2.5-Coder-7B Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Math-7B"

EXP_NAME=sensi_qwen25
RESULT_DIR=$RESULT_BASE_DIR/$EXP_NAME
mkdir -p $RESULT_DIR/data

OLD_DIR=$(pwd)
cd $RESULT_DIR
if [ ! -d pdfs ]; then
    mkdir -p pdfs
fi
$OLD_DIR/../data-vis/plot-variant-sensi.R data/

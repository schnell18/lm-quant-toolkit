#!/bin/bash

# export HF_HOME=/data/hugginface

EXP_RESULT_BASE_DIR=/fdata/llm/ieee-tai
if [ ! -d $EXP_RESULT_BASE_DIR/logs ]; then
    mkdir $EXP_RESULT_BASE_DIR/logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../../src/cli.py llm \
    --task eval_ppl \
    --model Qwen/Qwen3.5-4B \
    --algo fp16 \
    --experiment-name fix_ppl-qwen35-4b-fp16 \
    --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
    --result-dir="$EXP_RESULT_BASE_DIR/results" \
    2>&1 \
    | tee "$EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log"

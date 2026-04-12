#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

EXP_RESULT_BASE_DIR=/fdata/llm/ieee-tai
if [ ! -d $EXP_RESULT_BASE_DIR/logs ]; then
    mkdir $EXP_RESULT_BASE_DIR/logs
fi


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../../src/cli.py llm \
    --task eval_ppl \
    --model Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B \
    --algo bnb \
    --config b4g64 b8g128 \
    --experiment-name eval_ppl-bnb2 \
    --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots2" \
    --result-dir="$EXP_RESULT_BASE_DIR/results" \
    2>&1 \
    | tee "$EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log"

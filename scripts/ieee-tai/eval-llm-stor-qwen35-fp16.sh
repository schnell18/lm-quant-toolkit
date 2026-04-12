#!/bin/bash

# export HF_HOME=/data/hugginface

EXP_RESULT_BASE_DIR=/fdata/llm/ieee-tai
if [ ! -d $EXP_RESULT_BASE_DIR/logs ]; then
    mkdir $EXP_RESULT_BASE_DIR/logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=fp16
model_ids="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
for m in $model_ids; do
    python ../../src/cli.py llm \
        --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
        --result-dir="$EXP_RESULT_BASE_DIR/results" \
        --model $m \
        --algo ${algo} \
        --task eval_model_storage \
        --experiment-name eval_model_stor_fp16 \
        2>&1 \
        | tee $EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log
done

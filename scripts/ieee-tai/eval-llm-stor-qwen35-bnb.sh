#!/bin/bash

# export HF_HOME=/data/hugginface

EXP_RESULT_BASE_DIR=/fdata/llm/ieee-tai
if [ ! -d $EXP_RESULT_BASE_DIR/logs ]; then
    mkdir $EXP_RESULT_BASE_DIR/logs
fi


export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=bnb
model_ids="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
cfgs="b4g64 b8g128"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../../src/cli.py llm \
            --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots2" \
            --result-dir="$EXP_RESULT_BASE_DIR/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_stor-bnb \
            2>&1 \
            | tee "$EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log"
    done
done

#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task quant \
    --model Qwen/Qwen3.5-9B \
    --algo awq \
    --config b4g32 b4g64 b4g128 \
    --experiment-name quant_llm-qwen35-9b-awq \
    --quant-snapshot-dir="/fdata/llm/ieee-tai/snapshots" \
    --result-dir="/fdata/llm/ieee-tai/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

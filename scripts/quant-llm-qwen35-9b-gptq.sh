#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task quant \
    --model Qwen/Qwen3.5-9B \
    --algo gptq \
    --experiment-name quant_llm-qwen35-9b-gptq \
    --quant-snapshot-dir="/fdata/llm/ieee-tai/snapshots" \
    --result-dir="/fdata/llm/ieee-tai/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task quant \
    --model 0 1 2 \
    --algo gptq \
    --config b8g32 b8g64 b8g128 \
    --experiment-name quant_llm-8bit-gptq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

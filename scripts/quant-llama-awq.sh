#!/bin/bash

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/lm-quant-toolkit/src/cli.py llm \
    --task quant \
    --model 0 2 \
    --algo awq \
    --config b4g32 b4g128 \
    --experiment-name quant_llama-b4g32-128-awq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# MXQ requires Frobenius-norm metadata for Qwen3.5-9B.
# Generate it first if not already present:
#   python ../src/dump.py fnorm --model Qwen/Qwen3.5-9B --output-dir ../src/lm_quant_toolkit/data

python ../src/cli.py llm \
    --task quant \
    --model Qwen/Qwen3.5-9B \
    --algo mxq \
    --experiment-name quant_llm-qwen35-9b-mxq \
    --quant-snapshot-dir="/fdata/llm/ieee-tai/snapshots" \
    --result-dir="/fdata/llm/ieee-tai/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

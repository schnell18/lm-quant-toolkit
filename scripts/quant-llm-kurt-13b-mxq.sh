#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task quant \
    --model 1 \
    --algo mxq \
    --config 4.51 4.25 4.13 \
    --experiment-name quant_llm-weighted-113b-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots-kurt" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

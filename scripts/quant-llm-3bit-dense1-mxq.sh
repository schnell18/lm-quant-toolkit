#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task quant \
    --model 0 1 2 \
    --algo mxq \
    --config 3.57 3.55 3.53 3.51 3.49 3.47 3.45 3.31 3.29 3.27 3.25 3.23 3.21 3.19 3.17 3.15 3.13 3.11 3.09 3.07 \
    --experiment-name quant_llm_3bit_dense1-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

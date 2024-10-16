#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task eval_ppl \
    --model 0 1 2 \
    --algo mxq \
    --config 2.57 2.55 2.53 2.51 2.49 2.47 2.45 2.31 2.29 2.27 2.25 2.23 2.21 2.19 2.17 2.15 2.13 \
    --experiment-name eval_ppl_2bit_dense1-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log


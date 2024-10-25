#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task eval_ppl \
    --model 0 2 \
    --algo mxq \
    --config 3.51 3.25 3.13 4.51 4.25 4.13 \
    --experiment-name eval_ppl-kurt-7-8b-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots-kurt" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

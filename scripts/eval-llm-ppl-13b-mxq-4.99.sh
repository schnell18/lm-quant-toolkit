#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task eval_ppl \
    --model 1 \
    --algo mxq \
    --config 4.97 \
    --experiment-name eval_ppl_13b-mxq-ks-4_99 \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots/kurt-scaled" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

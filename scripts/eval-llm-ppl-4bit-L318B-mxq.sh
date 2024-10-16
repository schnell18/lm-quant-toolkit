#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task eval_ppl \
    --model 3 \
    --algo mxq \
    --config 4.61 4.59 4.57 4.55 4.53 4.51 4.49 4.47 4.45 4.43 4.41 4.35 4.33 4.31 4.29 4.27 4.25 4.23 4.21 4.19 4.17 4.15 4.13 4.11 4.09 4.07 4.05 4.03 \
    --experiment-name eval_ppl_4bit_L318B-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log


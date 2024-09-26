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
    --algo awq \
    --config b4g32 b4g64 b4g128 \
    --experiment-name eval_ppl_7B_8B-awq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

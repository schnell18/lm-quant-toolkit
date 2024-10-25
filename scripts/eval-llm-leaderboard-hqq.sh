#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python ../src/cli.py llm \
    --task eval_leaderboard \
    --model 0 1 2 \
    --algo hqq \
    --config b4g32 b4g64 b4g128 b3g32 b3g64 b3g128 \
    --experiment-name eval_leaderboard-hqq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

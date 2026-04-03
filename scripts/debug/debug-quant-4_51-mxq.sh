#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python -m pdb ../src/cli.py llm \
    --task quant \
    --model 0 \
    --algo mxq \
    --config 4.51 \
    --experiment-name debug_quant_llm_4_51-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots-debug" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

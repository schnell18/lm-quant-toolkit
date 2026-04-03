#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

    # --model 0 1 2 \
python -m pdb ../../src/cli.py llm \
    --task quant \
    --model 0  \
    --algo hqq \
    --experiment-name debug-quant_llm-hqq \
    --quant-snapshot-dir="/fdata/llm/ieee-tai/snapshots" \
    --result-dir="/fdata/llm/ieee-tai/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

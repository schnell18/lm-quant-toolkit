#!/bin/bash

# export HF_HOME=/data/hugginface

if [ ! -d logs ]; then
    mkdir logs
fi

python -m pdb ../../src/cli.py llm \
    --task quant \
    --model 0 \
    --algo awq \
    --config b4g64 \
    --experiment-name debug_quant_llm_awq-gm \
    --quant-snapshot-dir="/fdata/ieee-tai/snapshots-debug" \
    --result-dir="/fdata/ieee-tai/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

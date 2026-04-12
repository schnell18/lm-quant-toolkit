#!/bin/bash

# export HF_HOME=/data/hugginface

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=bnb
model_ids="0 1 2"
cfgs="b4g64 b8g128"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../../src/cli.py llm \
            --quant-snapshot-dir="/fdata/llm/ieee-tai/snapshots2" \
            --result-dir="/fdata/llm/ieee-tai/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_storge-bnb \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

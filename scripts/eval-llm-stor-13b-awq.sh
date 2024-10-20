#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=awq
model_ids="1"
cfgs="b4g32 b4g64 b4g128"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_stor_13b_awq \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done



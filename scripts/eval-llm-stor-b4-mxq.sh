#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=mxq
model_ids="0 1 2"
cfgs="4.51 4.25 4.13 3.51 3.25 3.13 3.02 2.51 2.25"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_stor_b4-mxq \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=bnb
model_ids="1"
cfgs="b4g64"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python -m pdb ../src/cli.py llm \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name debug_model_storge-bnb2 \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done


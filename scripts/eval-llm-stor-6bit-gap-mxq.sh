#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=mxq
model_ids="0 1 2"

cfgs="6.03 6.05 6.07 6.09 6.11 6.13 6.15 6.17 6.19 6.21 6.23 6.25 6.27 6.29 6.31 6.33 6.35 6.37 6.39 6.41 6.43 6.45 6.47 6.49 6.51 6.53 6.55 6.57 6.59 6.61 6.63 6.65 6.68 6.69"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_stor_dense-6bit-gap-mxq \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log

        find /fdata/llm/mxq/snapshots/meta-llama/ -maxdepth 1 -type d -cmin -3 | xargs rm -fr
    done
done



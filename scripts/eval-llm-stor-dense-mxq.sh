#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=mxq
model_ids="0 1 2"

cfgs="8.51 8.25 8.13 7.76 7.74 7.72 7.70 7.68 7.66 7.63 7.62 7.60 7.57 7.56 7.54 7.53 7.50 7.48 7.47 7.44 7.42 7.41 7.38 7.36 7.34 7.32 7.30 7.29 7.26 7.24 7.22 7.20 7.19 7.16 7.14 7.13 7.10 7.08 7.06 7.04 7.02 7.01 6.98 6.96 6.95 6.92 6.89 6.87 6.86 6.83 6.81 6.78 6.77 6.75 6.72 6.71 6.06 6.04 6.02 6.00 5.98 5.96 5.94 5.92 5.90 5.88 5.86 5.84 5.82 5.80 5.78 5.76 5.74 5.72 5.70 5.68 5.66 5.64 5.62 5.60 5.58 5.56 5.54 5.52 5.50 5.48 5.46 5.44 5.42 5.40 5.38 5.36 5.34 5.32 5.30 5.28 5.26 5.24 5.22 5.20 5.18 5.16 5.14 5.12 5.10 5.08 5.06 5.04 5.02 5.00 4.99 4.97 4.95 4.93 4.91 4.89 4.87 4.85 4.83 4.81 4.79 4.77 4.75 4.73 4.71 4.69 4.67 4.65 4.63 4.61 4.59 4.57 4.55 4.53 4.51 4.49 4.47 4.45 4.43 4.41 4.39 4.37 4.35 4.33 4.31 4.29 4.27 4.25 4.23 4.21 4.19 4.17 4.15 4.13 4.11 4.09 4.07 4.05 4.03 4.01 3.99 3.97 3.95 3.93 3.91 3.89 3.87 3.85 3.83 3.81 3.79 3.77 3.75 3.73 3.71 3.69 3.67 3.65 3.63 3.61 3.59 3.57 3.55 3.53 3.51 3.49 3.47 3.45 3.43 3.42 3.41 3.39 3.37 3.35 3.33 3.31 3.29 3.27 3.25 3.23 3.21 3.19 3.17 3.15 3.13 3.11 3.09 3.07"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_stor_dense2-mxq \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log

        find /fdata/llm/mxq/snapshots/meta-llama/ -maxdepth 1 -type d -cmin -3 | xargs rm -fr
    done
done



#!/bin/bash

MXQ_CONFIGS="2.13 2.15 2.17 2.19 2.21 2.23 2.25 2.27 2.29 2.31 2.45 2.47 2.49 2.51 2.53 2.55 2.57 3.07 3.09 3.11 3.13 3.15 3.17 3.19 3.21 3.23 3.25 3.27 3.29 3.31 3.33 3.35 3.37 3.39 3.41 3.42 3.43 3.45 3.47 3.49 3.51 3.53 3.55 3.57 3.59 3.61 3.63 3.65 3.67 3.69 3.71 3.73 3.75 3.77 3.79 3.81 3.83 3.85 3.87 3.89 3.91 3.93 3.95 3.97 3.99 4.01 4.03 4.05 4.07 4.09 4.11 4.13 4.15 4.17 4.19 4.21 4.23 4.25 4.27 4.29 4.31 4.33 4.35 4.37 4.39 4.41 4.43 4.45 4.47 4.49 4.51 4.53 4.55 4.57 4.59 4.61 4.63 4.65 4.67 4.69 4.71 4.73 4.75 4.77 4.79 4.81 4.83 4.85 4.87 4.89 4.91 4.93 4.95 4.97 4.99 5 5.02 5.04 5.06 5.08 5.1 5.12 5.14 5.16 5.18 5.2 5.22 5.24 5.26 5.28 5.3 5.32 5.34 5.36 5.38 5.4 5.42 5.44 5.46 5.48 5.5 5.52 5.54 5.56 5.58 5.6 5.62 5.64 5.66 5.68 5.7 5.72 5.74 5.76 5.78 5.8 5.82 5.84 5.86 5.88 5.9 5.92 5.94 5.96 5.98 6 6.02 6.03 6.04 6.05 6.06 6.07 6.09 6.11 6.13 6.15 6.17 6.19 6.21 6.23 6.25 6.27 6.29 6.31 6.33 6.35 6.37 6.39 6.41 6.43 6.45 6.47 6.49 6.51 6.53 6.55 6.57 6.59 6.61 6.63 6.65 6.68 6.69 6.71 6.72 6.75 6.77 6.78 6.81 6.83 6.86 6.87 6.89 6.92 6.95 6.96 6.98 7.01 7.02 7.04 7.06 7.08 7.1 7.13 7.14 7.16 7.19 7.2 7.22 7.24 7.26 7.29 7.3 7.32 7.34 7.36 7.38 7.41 7.42 7.44 7.47 7.48 7.5 7.53 7.54 7.56 7.57 7.6 7.62 7.63 7.66 7.68 7.7 7.72 7.74 7.76 8.13 8.25 8.51"
EXP_NAME="kurt-scaled-dense"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots/kurt-scaled"
python ../src/cli.py llm \
    --task eval_ppl \
    --model 0 1 2 \
    --algo mxq \
    --config $MXQ_CONFIGS \
    --experiment-name "${EXP_NAME}_ppl" \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

python ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --budget $MXQ_CONFIGS \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
    --output_file mxq-quant-cfgs-kurt-scaled-dense.csv

algo=mxq
model_ids="0 1 2"
for m in $model_ids; do
    for cfg in $MXQ_CONFIGS; do
        python ../src/cli.py llm \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name "${EXP_NAME}_stor" \
            --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
            --result-dir="/fdata/llm/mxq/results" \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

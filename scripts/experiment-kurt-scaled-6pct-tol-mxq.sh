#!/bin/bash

EXP_NAME="kurt-scaled-6pct-sol"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots-${EXP_NAME}"
python ../src/cli.py llm \
    --task eval_ppl \
    --model 0 1 2 \
    --algo mxq \
    --config 3.51 3.25 3.13 4.51 4.25 4.13 \
    --experiment-name "${EXP_NAME}_ppl" \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

python ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --budget 3.51 3.25 3.13 4.51 4.25 4.13 \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
    --output_file mxq-quant-cfgs-kurt-scaled-6pct-tol.csv

algo=mxq
model_ids="0 1 2"
cfgs="3.51 3.25 3.13 4.51 4.25 4.13"
for m in $model_ids; do
    for cfg in $cfgs ; do
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

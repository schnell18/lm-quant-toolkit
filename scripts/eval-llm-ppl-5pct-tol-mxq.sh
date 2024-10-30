#!/bin/bash

# python ../src/cli.py llm \
#     --task eval_ppl \
#     --model 0 1 2 \
#     --algo mxq \
#     --config 3.51 3.25 3.13 4.51 4.25 4.13 \
#     --experiment-name quant-mxq \
#     --quant-snapshot-dir="/fdata/llm/mxq/snapshots-5pct-tol" \
#     --result-dir="/fdata/llm/mxq/results" \
#     2>&1 \
#     | tee logs/bench-$(date +%Y%m%d%H%M%S).log
#
# python ../src/cli.py dump \
#     --type quant_config \
#     --model 0 1 2 \
#     --budget 3.51 3.25 3.13 4.51 4.25 4.13 \
#     --quant-snapshot-dir="/fdata/llm/mxq/snapshots-5pct-tol" \
#     --output_file mxq-quant-cfgs-mxq1-5pct-tol.csv

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
            --experiment-name eval_stor_pct5-mxq \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots-5pct-tol" \
            --result-dir="/fdata/llm/mxq/results" \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done



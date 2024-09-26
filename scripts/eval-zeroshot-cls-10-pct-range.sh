#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

python ../src/cli.py vit \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    --model 0 1 2 \
    --config 8.05 7.97 7.89 7.80 7.72 7.64 7.56 7.48 7.40 7.32 4.46 4.42 4.37 4.33 4.28 4.24 4.19 4.15 4.10 4.06 3.47 3.44 3.40 3.37 3.33 3.30 3.26 3.23 3.19 3.16 2.99 2.96 2.93 2.90 2.87 2.84 2.81 2.78 2.75 2.72 \
    --algo mxq \
    --task eval_zeroshot_cls \
    --experiment-name eval_zs_H_mxq_10_pct_range \
    2>&1 \
    | tee logs/bench-vit-$(date +%Y%m%d%H%M%S).log


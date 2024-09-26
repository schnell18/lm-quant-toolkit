#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

python ../src/cli.py vit \
    --task eval_linear_probe \
    --model 0 1 2 \
    --algo hqq \
    --experiment-name eval_lp_BLH_hqq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-vit-$(date +%Y%m%d%H%M%S).log

#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

python ../src/cli.py vit \
    --task eval_linear_probe \
    --model 0 1 \
    --algo mxq \
    --config 2.51 2.46 2.25 2.20 \
    --experiment-name eval_lp_BH_b2_4-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-vit-$(date +%Y%m%d%H%M%S).log



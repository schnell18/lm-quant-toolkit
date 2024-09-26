#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

python ../src/cli.py vit \
    --task eval_zeroshot_cls \
    --model 0 1 2 \
    --algo mxq \
    --experiment-name eval_zs_BLH_mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-vit-$(date +%Y%m%d%H%M%S).log


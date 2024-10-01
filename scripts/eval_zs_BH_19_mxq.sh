#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

python ../src/cli.py vit \
    --task eval_zeroshot_cls \
    --model 0 1 \
    --algo mxq \
    --config 7.80 7.72 7.64 7.56 7.48 5.00 4.95 4.73 4.46 4.42 4.37 4.33 4.28 3.40 3.33 3.23 3.19 3.16 2.99 2.96 \
    --experiment-name eval_zs_BH_19_mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-vit-$(date +%Y%m%d%H%M%S).log



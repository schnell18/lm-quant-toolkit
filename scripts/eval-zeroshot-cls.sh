#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

python /home/justin/work/lm-quant-toolkit/src/cli.py vit \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    --model 0 1 \
    --algo mxq \
    --task eval_zeroshot_cls \
    --experiment-name eval_zs_BH_mxq \
    2>&1 \
    | tee logs/bench-vit-$(date +%Y%m%d%H%M%S).log


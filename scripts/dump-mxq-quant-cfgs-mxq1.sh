#!/bin/bash

python ../src/cli.py llm \
    --task quant \
    --model 0 1 2 \
    --algo mxq \
    --config 3.51 3.25 3.13 4.51 4.25 4.13 \
    --experiment-name quant-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

python ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --budget 3.51 3.25 3.13 4.51 4.25 4.13 \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --output_file mxq-quant-cfgs-mxq1.csv

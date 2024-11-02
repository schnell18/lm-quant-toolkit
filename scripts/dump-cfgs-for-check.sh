#!/bin/bash

python ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --attempt mxq1 kurt-global kurt-scaled pct5 pct6 \
    --budget 3.51 3.25 3.13 4.51 4.25 4.13 \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --output_file mxq-mem-bound-check.csv

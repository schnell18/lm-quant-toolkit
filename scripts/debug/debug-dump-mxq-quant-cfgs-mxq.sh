#!/bin/bash

python -m pdb ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --budget 3.51 3.25 3.13 4.51 4.25 4.13 \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots-kurt-scaled-6pct-sol/" \
    --output_file mxq-quant-cfgs-mxq1.csv

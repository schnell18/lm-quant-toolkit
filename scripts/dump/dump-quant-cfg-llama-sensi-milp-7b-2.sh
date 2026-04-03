#!/bin/bash

ATTEMPT="llama-sensi-milp-7b-2"
python ../src/cli.py dump \
    --type quant_config \
    --model 0 \
    --attempt $ATTEMPT\
    --budget 4.51 \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --output-file ${ATTEMPT}.csv


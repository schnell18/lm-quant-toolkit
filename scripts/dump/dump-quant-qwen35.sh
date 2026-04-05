#!/bin/bash

CONFIGS="b4g64"
QUANT_SNAPSHOT_BASE_DIR="/home/justin/work/hqq-pristine"
ATTEMPT=hqq
python -m pdb ../../src/cli.py dump \
    --type quant_config \
    --model Qwen/Qwen3.5-9B \
    --budget $CONFIGS \
    --attempt $ATTEMPT \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_BASE_DIR \
    --output-file quant-cfg-allot-hqq.csv

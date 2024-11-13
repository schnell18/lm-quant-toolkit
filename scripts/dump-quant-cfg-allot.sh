#!/bin/bash

CONFIGS="2.13 2.25 2.51 3.51 3.25 3.13 4.51 4.25 4.13"
QUANT_SNAPSHOT_BASE_DIR="/fdata/llm/mxq/snapshots"

ATTEMPT="mxq1"
QUANT_SNAPSHOT_DIR="$QUANT_SNAPSHOT_BASE_DIR/$ATTEMPT"
python ../src/cli.py llm \
    --task quant \
    --model 0 1 2 \
    --algo mxq \
    --config $CONFIGS \
    --experiment-name quant-mxq-allot-$ATTEMPT \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

python ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --budget $CONFIGS \
    --attempt $ATTEMPT \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_BASE_DIR \
    --output_file quant-cfg-allot-$ATTEMPT.csv

ATTEMPT="kurt-scaled"
QUANT_SNAPSHOT_DIR="$QUANT_SNAPSHOT_BASE_DIR/$ATTEMPT"
python ../src/cli.py llm \
    --task quant \
    --model 0 1 2 \
    --algo mxq \
    --weight-algo $ATTEMPT \
    --config $CONFIGS \
    --experiment-name quant-mxq-allot-$ATTEMPT \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log


python ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --budget $CONFIGS \
    --attempt $ATTEMPT \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_BASE_DIR \
    --output_file quant-cfg-allot-$ATTEMPT.csv


CONFIGS="b2g64 b2g32 b3g128 b3g64 b3g32 b4g128 b4g64 b4g32"
QUANT_SNAPSHOT_DIR="$QUANT_SNAPSHOT_BASE_DIR/hqq"
python ../src/cli.py llm \
    --task quant \
    --model 0 1 2 \
    --algo hqq \
    --config $CONFIGS \
    --experiment-name quant-hqq-allot \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log


ATTEMPT=hqq
python ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --budget $CONFIGS \
    --attempt $ATTEMPT \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_BASE_DIR \
    --output_file quant-cfg-allot-hqq.csv

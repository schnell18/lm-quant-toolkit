#!/bin/bash

CFG=b4g64
DS=pileval
MODEL=1
OUT_FILE="debug-13b.csv"

python -m pdb ../src/cli.py dump \
    --type sensitivity \
    --model $MODEL \
    --config $CFG \
    --calib-dataset $DS \
    --output-file $OUT_FILE

#!/bin/bash

export USE_CPU_FOR_SENSITIVITY=0
python ../src/cli.py dump \
    --type sensitivity \
    --model 1 \
    --output-file llama2-13b-sensitivity.csv

#!/bin/bash

python ../src/cli.py dump \
    --type sensitivity \
    --model 0 2 \
    --output-file llama-sensitivity.csv

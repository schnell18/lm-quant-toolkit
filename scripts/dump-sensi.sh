#!/bin/bash

CALIB_DATASETS="bos pileval wikitext c4"
#CONFIGS="b3g128 b3g64 b3g32 b4g128 b4g64 b4g32 b8g128 b8g64 b8g32"
CONFIGS="b2g128 b2g64 b2g32"
MODELS="0 2"

for DS in $CALIB_DATASETS; do
    for CFG in $CONFIGS; do
        for MODEL in $MODELS; do
            OUT_FILE="llama-sensitivity-${MODEL}-${CFG}-${DS}.csv"
            python ../src/cli.py dump \
                --type sensitivity \
                --model $MODEL \
                --config $CFG \
                --calib-dataset $DS \
                --output-file $OUT_FILE
        done
    done
done

#!/bin/bash

CALIB_DATASETS="pileval wikitext c4"
CONFIGS="b3g128 b3g64 b4g128 b4g64 b8g128 b8g64"
MODELS="meta-llama/Llama-2-7b-hf meta-llama/Llama-2-7b-chat-hf epfl-llm/meditron-7b meta-llama/Llama-3.1-8B meta-llama/Llama-3.1-8B-Instruct"

for DS in $CALIB_DATASETS; do
    for CFG in $CONFIGS; do
        for MODEL in $MODELS; do
            OUT_FILE="variant-sensi-${MODEL}-${CFG}-${DS}.csv"
            python ../src/dump.py dump \
                --type sensitivity \
                --model $MODEL \
                --config $CFG \
                --calib-dataset $DS \
                --output-file $OUT_FILE
        done
    done
done

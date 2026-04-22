#!/bin/bash

MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"

mkdir -p /tmp/kurtosis-dump
python ../../src/dump.py kurtosis \
    --model $MODELS \
    --output-dir /tmp/kurtosis-dump

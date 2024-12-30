#!/bin/bash

MODELS="meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Meta-Llama-3-8B"

# meta-llama/Llama-2-7b-hf
# meta-llama/Llama-2-13b-hf
# meta-llama/Meta-Llama-3-8B
# meta-llama/Meta-Llama-3-70B
# meta-llama/Llama-2-70b-hf
# meta-llama/Meta-Llama-3-70B-Instruct
# meta-llama/Meta-Llama-3.1-405B-Instruct

mkdir -p /tmp/kurtosis-dump
python ../src/dump.py kurtosis \
    --model $MODELS \
    --output-dir /tmp/kurtosis-dump

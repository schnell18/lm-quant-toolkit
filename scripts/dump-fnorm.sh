#!/bin/bash

MODELS="meta-llama/Meta-Llama-3-8B"
#MODELS="meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Meta-Llama-3-8B"

# meta-llama/Llama-2-7b-hf
# meta-llama/Llama-2-13b-hf
# meta-llama/Meta-Llama-3-8B
# meta-llama/Meta-Llama-3-70B
# meta-llama/Llama-2-70b-hf
# meta-llama/Meta-Llama-3-70B-Instruct
# meta-llama/Meta-Llama-3.1-405B-Instruct

mkdir -p /tmp/fnorm-dump
python ../src/dump.py fnorm \
    --model $MODELS \
    --output-dir /tmp/fnorm-dump

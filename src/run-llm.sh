#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export HF_HOME=/home/justin/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python ../src/lm_quant_toolkit/eval/bench.py 2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

export HF_HOME=/home/justin/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/justin/work/lm-quant-toolkit/src/cli.py llm \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    --model 1 \
    --algo hqq \
    --config b3g32 \
    --task quant \
    --experiment-name quant_llama-2-13b-b3g32-hqq \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

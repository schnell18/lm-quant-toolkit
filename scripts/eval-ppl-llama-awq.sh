#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

export HF_HOME=/home/justin/.cache/huggingface/
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python /home/justin/work/lm-quant-toolkit/src/cli.py llm \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    --model 0 \
    --algo awq \
    --config b4g32 b4g64 b4g128 \
    --task eval_ppl \
    --experiment-name eval_ppl_llama2-7b-b4-awq \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

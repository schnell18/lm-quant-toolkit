#!/bin/bash

# export HF_HOME=/data/hugginface

EXP_RESULT_BASE_DIR=/fdata/llm/ieee-tai/awq-gptq-retest
if [ ! -d $EXP_RESULT_BASE_DIR/logs ]; then
    mkdir $EXP_RESULT_BASE_DIR/logs
fi
if [ ! -d $EXP_RESULT_BASE_DIR/snapshots ]; then
    mkdir $EXP_RESULT_BASE_DIR/snapshots
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# mkdir -p $EXP_RESULT_BASE_DIR/results-awq
# python ../../src/cli.py llm \
#     --task eval_ppl \
#     --model Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B \
#     --algo awq \
#     --config b4g32 b4g64 b4g128 \
#     --experiment-name eval_ppl-qwen35-awq \
#     --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
#     --result-dir="$EXP_RESULT_BASE_DIR/results-awq" \
#     2>&1 \
#     | tee "$EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log"

mkdir -p $EXP_RESULT_BASE_DIR/results-gptq
python ../../src/cli.py llm \
    --task eval_ppl \
    --model Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B \
    --algo gptq \
    --experiment-name eval_ppl-qwen35-gptq \
    --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
    --result-dir="$EXP_RESULT_BASE_DIR/results-gptq" \
    2>&1 \
    | tee "$EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log"

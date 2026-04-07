#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

EXP_RESULT_BASE_DIR=/fdata/llm/ieee-tai
if [ ! -d $EXP_RESULT_BASE_DIR/logs ]; then
    mkdir $EXP_RESULT_BASE_DIR/logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# first pass: quantize the models
python ../../src/cli.py llm \
    --task quant \
    --model Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B \
    --algo gptq \
    --experiment-name quant-qwen35-gptq \
    --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
    --result-dir="$EXP_RESULT_BASE_DIR/results" \
    2>&1 \
    | tee "$EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log"

# second pass: evaluate PPL
python ../../src/cli.py llm \
    --task eval_ppl \
    --model Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B \
    --algo gptq \
    --experiment-name eval_ppl-qwen35-gptq \
    --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
    --result-dir="$EXP_RESULT_BASE_DIR/results" \
    2>&1 \
    | tee "$EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log"

#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

EXP_RESULT_BASE_DIR=/fdata/llm/ieee-tai/gptq-correction
if [ ! -d $EXP_RESULT_BASE_DIR/logs ]; then
    mkdir $EXP_RESULT_BASE_DIR/logs
fi
if [ ! -d $EXP_RESULT_BASE_DIR/snapshots ]; then
    mkdir $EXP_RESULT_BASE_DIR/snapshots
fi
if [ ! -d $EXP_RESULT_BASE_DIR/results ]; then
    mkdir $EXP_RESULT_BASE_DIR/results
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


# second pass: evaluate memory
python ../../src/cli.py llm \
    --task eval_model_storage \
    --model Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B \
    --algo gptq \
    --experiment-name eval_mem-qwen35-gptq \
    --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
    --result-dir="$EXP_RESULT_BASE_DIR/results" \
    2>&1 \
    | tee "$EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log"

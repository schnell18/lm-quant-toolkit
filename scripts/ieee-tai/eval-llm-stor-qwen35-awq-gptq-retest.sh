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
# algo=awq
# model_ids="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
# cfgs="b4g32 b4g64 b4g128"
# for m in $model_ids; do
#     for cfg in $cfgs ; do
#         python ../../src/cli.py llm \
#             --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
#             --result-dir="$EXP_RESULT_BASE_DIR/results-awq" \
#             --model $m \
#             --algo ${algo} \
#             --config ${cfg} \
#             --task eval_model_storage \
#             --experiment-name eval_model_stor_awq \
#             2>&1 \
#             | tee $EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log
#     done
# done

mkdir -p $EXP_RESULT_BASE_DIR/results-gptq
algo=gptq
model_ids="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
cfgs="b8g32 b8g64 b8g128 b4g32 b4g64 b4g128 b3g32 b3g64 b3g128"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../../src/cli.py llm \
            --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots" \
            --result-dir="$EXP_RESULT_BASE_DIR/results-gptq" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_mem_gptq3 \
            2>&1 \
            | tee $EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

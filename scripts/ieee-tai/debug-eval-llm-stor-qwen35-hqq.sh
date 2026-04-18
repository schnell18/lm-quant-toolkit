#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

EXP_RESULT_BASE_DIR=/fdata/llm/ieee-tai
if [ ! -d $EXP_RESULT_BASE_DIR/logs ]; then
    mkdir $EXP_RESULT_BASE_DIR/logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=hqq
model_ids="Qwen/Qwen3.5-9B"
cfgs="b4g64"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python -m pdb ../../src/cli.py llm \
            --quant-snapshot-dir="$EXP_RESULT_BASE_DIR/snapshots2" \
            --result-dir="$EXP_RESULT_BASE_DIR/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name debug-eval_model_stor_hqq \
            2>&1 \
            | tee $EXP_RESULT_BASE_DIR/logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

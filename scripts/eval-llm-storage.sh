#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

algo=fp16
model_ids="0 1 2"
cfgs="base"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_storge_${algo}_${m}_${cfg} \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

algo=hqq
model_ids="0 1 2"
cfgs="b4g32 b4g64 b4g128 b3g32 b3g64 b3g128 b2g16 b2g32 b2g64"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_storge_${algo}_${m}_${cfg} \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

algo=mxq
model_ids="0 1 2"
cfgs="5_00 4_75 4_50 4_25 4_01 3_76 3_50 3_00 2_75 2_48"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_storge_${algo}_${m}_${cfg} \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

algo=gptq
model_ids="0 1 2"
cfgs="b4g32 b4g64 b4g128 b3g32 b3g64 b3g128"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_storge_${algo}_${m}_${cfg} \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done


algo=awq
model_ids="0 1 2"
cfgs="b4g32 b4g64 b4g128"
for m in $model_ids; do
    for cfg in $cfgs ; do
        python ../src/cli.py llm \
            --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
            --result-dir="/fdata/llm/mxq/results" \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name eval_model_storge_${algo}_${m}_${cfg} \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done



#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

if [ ! -d logs ]; then
    mkdir logs
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True


#  7.76 7.74 7.72 7.70
#  7.68 7.66 7.63 7.62
#  7.60 7.57 7.56

#  6.92 6.89 6.87 6.86
#  6.83 6.81 6.78 6.77
#  6.75 6.72 6.71 6.06
#  6.04 6.02 6.00

#  5.98 5.96 5.94 5.92
#  5.90 5.88 5.86 5.20
#  5.18 5.16 5.14 5.12
#  5.10 5.08 5.06 5.04
#  5.02 5.00

python ../src/cli.py llm \
    --task eval_ppl \
    --model 0 1 2 \
    --algo mxq \
    --config 7.76 7.74 7.72 7.70 7.68 7.66 7.63 7.62 7.60 7.57 7.56 6.92 6.89 6.87 6.86 6.83 6.81 6.78 6.77 6.75 6.72 6.71 6.06 6.04 6.02 6.00 5.98 5.96 5.94 5.92 5.90 5.88 5.86 5.20 5.18 5.16 5.14 5.12 5.10 5.08 5.06 5.04 5.02 5.00 \
    --experiment-name eval_ppl_567bit_dense1-mxq \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log



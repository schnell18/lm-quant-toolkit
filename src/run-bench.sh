#!/bin/bash

# export HF_HOME=/data/hugginface
# conda activate quant-eval

/home/justin/miniconda3/envs/quant-eval/bin/python \
    lm_quant_toolkit/eval/bench.py 2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

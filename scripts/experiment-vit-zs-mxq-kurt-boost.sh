#!/bin/bash

if [ ! -d logs ]; then
    mkdir logs
fi

# python ../src/cli.py vit \
#     --task eval_zeroshot_cls \
#     --model 0 1 \
#     --config 4.51 4.25 4.13 3.51 3.25 3.13\
#     --algo mxq \
#     --weight-algo kurt-boost \
#     --boost-stop 2 \
#     --top-m-layer 1 \
#     --experiment-name eval_zs_BH_mxq_kurt_boost \
#     --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
#     --result-dir="/fdata/llm/mxq/results" \
#     2>&1 \
#     | tee logs/bench-vit-$(date +%Y%m%d%H%M%S).log

python ../src/cli.py vit \
    --task eval_zeroshot_cls \
    --model 0 1 \
    --config 4.51 4.25 4.13 3.51 3.25 3.13\
    --algo mxq \
    --weight-algo kurt-boost \
    --boost-stop 2 \
    --top-m-layer 2 \
    --experiment-name eval_zs_BH_mxq_kurt_boost_22 \
    --quant-snapshot-dir="/fdata/llm/mxq/snapshots" \
    --result-dir="/fdata/llm/mxq/results" \
    2>&1 \
    | tee logs/bench-vit-$(date +%Y%m%d%H%M%S).log


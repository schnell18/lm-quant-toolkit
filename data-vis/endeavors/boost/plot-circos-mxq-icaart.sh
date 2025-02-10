#!/bin/bash
#

TOOLKIT_DIR="../../.."


# Other models:

MODELS="
Llama-2-7b-hf
Llama-2-13b-hf
Meta-Llama-3-8B
"
BUDGETS="4.25 4.13 4.51"

for MODEL in $MODELS; do
    for BUDGET in $BUDGETS; do
        $TOOLKIT_DIR/data-vis/plot-circos-allot-icaart.R \
          --model $MODEL \
          --budget $BUDGET \
          --fnorm_data_dir $TOOLKIT_DIR/src/data/ \
          --ppl_csv_file data/combined.csv \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv \
          --attempt1 hqq\
          --attempt2 mxq1
    done
done

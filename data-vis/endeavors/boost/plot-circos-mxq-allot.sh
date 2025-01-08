#!/bin/bash
#

TOOLKIT_DIR="../../.."


# Other models:

MODELS="
Llama-3-7b-hf
Llama-3-13b-hf
Meta-Llama-3-8B
"
BUDGETS="4.25 3.51"

STOP=2
TOPM=2
for MODEL in $MODELS; do
    for BUDGET in $BUDGETS; do
        $TOOLKIT_DIR/data-vis/plot-circos-allot.R \
          --model $MODEL \
          --budget $BUDGET \
          --fnorm_data_dir $TOOLKIT_DIR/src/data/ \
          --ppl_csv_file data/combined.csv \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv \
          --attempt1 sensi-boost-${STOP}-${TOPM} \
          --attempt2 kurt-boost-${STOP}-${TOPM} \
          --attempt3 hqq\
          --attempt4 mxq1
    done
done

#!/bin/bash
#

TOOLKIT_DIR="../../.."


# Other models:

MODELS="
Llama-2-7b-hf
Llama-2-13b-hf
Meta-Llama-3-8B
"
#BUDGETS="6.89 5.72 5.02 4.51 4.25 4.21 4.17 4.13 4.11 4.07 3.95 3.87 3.83 3.65 3.51 3.25 3.19 3.15 3.13 3.11 3.07"

BUDGETS="6.89 5.72 5.02"
if [[ ! -d pdfs/allot ]]; then
    mkdir -p pdfs/allot
fi

for MODEL in $MODELS; do
    for BUDGET in $BUDGETS; do
        $TOOLKIT_DIR/data-vis/plot-circos-allot.R \
          --model $MODEL \
          --budget $BUDGET \
          --fnorm_data_dir $TOOLKIT_DIR/src/data/ \
          --ppl_csv_file data/combined.csv \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv \
          --attempt1 kurt-milp-1 \
          --attempt2 kurt-milp-2 \
          --attempt3 kurt-milp-3 \
          --attempt4 kurt-milp-abl
    done
done

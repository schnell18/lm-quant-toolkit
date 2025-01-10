#!/bin/bash
#

TOOLKIT_DIR="../../.."


# Other models:

MODELS="
Llama-2-7b-hf
Llama-2-13b-hf
Meta-Llama-3-8B
"

BUDGETS="3.51 3.25 3.13"
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
          --attempt1 sensi-milp-1 \
          --attempt2 kurt-milp-1 \
          --attempt3 kurt-milp-2 \
          --attempt4 mxq2
    done
done


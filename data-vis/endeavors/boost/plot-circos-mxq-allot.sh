#!/bin/bash
#

TOOLKIT_DIR="../../.."


# Other models:

MODELS="
Llama-2-7b-hf
Llama-2-13b-hf
Meta-Llama-3-8B
"
BUDGETS="3.13 3.25 3.51 4.13 4.25 4.51"

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
          --attempt1 SensiBoost${STOP}${TOPM} \
          --attempt2 KurtBoost${STOP}${TOPM} \
          --attempt3 SensiAbl${STOP}${TOPM} \
          --attempt4 KurtAbl${STOP}${TOPM}
    done
done
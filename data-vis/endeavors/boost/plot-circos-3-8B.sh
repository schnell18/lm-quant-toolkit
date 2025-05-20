#!/bin/bash
#

TOOLKIT_DIR="../../.."


# Other models:

MODELS="
Meta-Llama-3-8B
"

BUDGETS="4.25"
if [[ ! -d pdfs/allot ]]; then
    mkdir -p pdfs/allot
fi

# for MODEL in $MODELS; do
#     for BUDGET in $BUDGETS; do
#         $TOOLKIT_DIR/data-vis/plot-circos-allot.R \
#           --model $MODEL \
#           --budget $BUDGET \
#           --fnorm_data_dir $TOOLKIT_DIR/src/data/ \
#           --ppl_csv_file data/combined.csv \
#           --quant_cfg_allot_file data/quant-cfg-allocation.csv \
#           --attempt1 sensi-boost-2-1 \
#           --attempt2 kurt-boost-2-1 \
#           --attempt3 hqq \
#           --attempt4 mxq1
#     done
# done
#
MODEL="Meta-Llama-3-8B"
BUDGET=4.13

$TOOLKIT_DIR/data-vis/plot-circos-allot.R \
  --model $MODEL \
  --budget $BUDGET \
  --fnorm_data_dir $TOOLKIT_DIR/src/data/ \
  --ppl_csv_file data/combined.csv \
  --quant_cfg_allot_file data/quant-cfg-allocation.csv \
  --attempt1 sensi-boost-2-2 \
  --attempt2 sensi-abl-2-2 \
  --attempt3 kurt-boost-2-2 \
  --attempt4 hqq

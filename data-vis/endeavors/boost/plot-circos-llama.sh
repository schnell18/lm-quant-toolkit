#!/bin/bash
#

TOOLKIT_DIR="../../.."
# MODEL="Meta-Llama-3-8B"
MODEL="Llama-2-13b-hf"
BUDGET=4.13

if [[ ! -d pdfs/allot ]]; then
    mkdir -p pdfs/allot
fi

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

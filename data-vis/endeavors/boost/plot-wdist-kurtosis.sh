#!/bin/bash
#
TOOLKIT_DIR="../../.."

MODELS="
Llama-2-13b-hf
Llama-2-7b-hf
Meta-Llama-3-8B
"

for MODEL in $MODELS; do
  $TOOLKIT_DIR/data-vis/plot-wdist-llm.R \
    --model_id $MODEL \
    --wdist_dir $TOOLKIT_DIR/data-vis/data/wdist
done

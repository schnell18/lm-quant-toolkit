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
        $TOOLKIT_DIR/data-vis/plot-mxq-allot-trio.R \
          --model $MODEL \
          --budget $BUDGET \
          --baseline_data_dir $TOOLKIT_DIR/src/data/ \
          --attempt1 sensi-boost-${STOP}-${TOPM} \
          --attempt2 kurt-boost-${STOP}-${TOPM} \
          --attempt3 sensi-abl-${STOP}-${TOPM}
    done
done

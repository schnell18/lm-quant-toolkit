#!/bin/bash

quantise() {
    echo "=========Quantise Model========="
    python ../../src/cli.py llm \
      --task quant \
      --model $MODELS \
      --algo mxq \
      --ablation \
      --weight-algo $weight_algo \
      --boost-stop $BOOST_STOP \
      --top-m-layer $BOOST_TOP_M \
      --config ${BUDGETS} \
      --experiment-name "${EXP_NAME}_qnt" \
      --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
      --result-dir=$RESULT_DIR \
      2>&1 \
      | tee -a $log_file
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
      echo "quantisation failed!"
      exit $EXIT_CODE
    fi
}

eval_ppl() {
    echo "=========Run perplexity evaluation========="
    python ../../src/cli.py llm \
      --task eval_ppl \
      --model $MODELS \
      --algo mxq \
      --weight-algo $weight_algo \
      --boost-stop $BOOST_STOP \
      --top-m-layer $BOOST_TOP_M \
      --config ${BUDGETS} \
      --experiment-name "${EXP_NAME}_ppl" \
      --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
      --result-dir=$RESULT_DIR \
      2>&1 \
      | tee -a $log_file
    EXIT_CODE=$?
    if [ $EXIT_CODE -ne 0 ]; then
      echo "Perplexity evaluation failed!"
      exit $EXIT_CODE
    fi
}

dump_qnt_cfg() {
    echo "=========Dump quantization configs on batch ${EXP_NAME}========="
    mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT
    python ../../src/cli.py dump \
      --type quant_config \
      --model $MODELS \
      --budget ${BUDGETS} \
      --attempt $ATTEMPT \
      --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
      --output-file "$RESULT_DIR/${EXP_NAME}_cfg/quant-allot-${EXP_NAME}.csv" \
      2>&1 \
      | tee -a $log_file
}

eval_stor() {
    echo "=========Run memory evaluation on batch ${EXP_NAME}========="
    algo=mxq
    model_ids=$MODELS
    for m in $model_ids; do
      for cfg in ${BUDGETS}; do
          python ../../src/cli.py llm \
              --model $m \
              --algo ${algo} \
              --config ${cfg} \
              --task eval_model_storage \
              --experiment-name "${EXP_NAME}_stor" \
              --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
              --result-dir=$RESULT_DIR \
              2>&1 \
              | tee -a $log_file
      done
    done
}

plot_results() {
    echo ""
    # OLD_DIR=$(pwd)
    # cd $RESULT_DIR/$EXP_BASE_NAME
    # if [ ! -d pdfs/allot ]; then
    #     mkdir -p pdfs/allot
    # fi
    # $OLD_DIR/../data-vis/combine.R \
    #     --baseline_data_dir $OLD_DIR/../data-vis/data \
    #     --mxq_data_dir data
    # $OLD_DIR/../../data-vis/plot-ppl-mem.R -d data/combined.csv
    # $OLD_DIR/../../data-vis/plot-mem-consumption.R data/combined.csv
    # $OLD_DIR/../../data-vis/plot-quant-speed.R data/combined.csv
    # $OLD_DIR/../../data-vis/gen-table-mxq-llm.R --csv_file data/combined.csv --attempt $ATTEMPT
    # pdflatex table.tex
    #
    # # plot configuration allocations for 3 * 12 MXQ combinations
    # for model in $MODEL_NAMES; do
    #     for budget in $BUDGETS; do
    #         $OLD_DIR/../data-vis/plot-mxq-allocation.R \
    #           -m $model \
    #           -b $budget \
    #           --attempt1 $ATTEMPT \
    #           --attempt2 mxq1 \
    #           --fnorm_data_dir $OLD_DIR/../src/data \
    #           --quant_cfg_allot_file data/quant-cfg-allocation.csv
    #     done
    # done
    # cd $OLD_DIR
}

BUDGETS="3.13 3.25 3.51 4.13 4.25 4.51"
LOG_DIR="/fdata/llm/ieee-tai/logs5"
RESULT_DIR="/fdata/llm/ieee-tai/results5"
QUANT_SNAPSHOT_DIR="/fdata/llm/ieee-tai/snapshots5"

# Use cached dataset to speedup wikitext, c4 ppl evaluation
# export HF_DATASETS_OFFLINE=1
weight_algo=kurt-boost
MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"
MODEL_NAMES="Qwen3.5-2B Qwen3.5-4B Qwen3.5-9B"
# MODELS="Qwen/Qwen3.5-9B"
# MODEL_NAMES="Qwen3.5-9B"


BOOST_STOPS="2 3"
BOOST_TOP_MS="1 2 3 0"
# BOOST_STOPS="2"
# BOOST_TOP_MS="1"

for BOOST_STOP in $BOOST_STOPS; do
    for BOOST_TOP_M in $BOOST_TOP_MS; do
        ATTEMPT="kurt-boost-${BOOST_STOP}-${BOOST_TOP_M}"
        EXP_BASE_NAME=$ATTEMPT
        EXP_NAME="${ATTEMPT}"
        mkdir -p $LOG_DIR
        mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT
        mkdir -p $RESULT_DIR/${EXP_NAME}_ppl
        mkdir -p $RESULT_DIR/${EXP_NAME}_cfg
        mkdir -p $RESULT_DIR/${EXP_NAME}_qnt
        mkdir -p $RESULT_DIR/${EXP_NAME}_stor

        log_file="$LOG_DIR/bench-${ATTEMPT}-$(date +%Y%m%d%H%M%S).log"

        quantise
        eval_stor
        dump_qnt_cfg
        # eval_ppl

    done
done

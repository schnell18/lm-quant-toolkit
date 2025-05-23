#!/bin/bash

BUDGETS="3.25"
RESULT_DIR="/fdata/llm/mxq/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"

# Use cached dataset to speedup wikitext, c4 ppl evaluation
export HF_DATASETS_OFFLINE=1
weight_algo=sensi-boost
MODELS="0"
MODEL_NAMES="Llama-2-7b-hf"


# BOOST_STOPS="2 3"
BOOST_STOPS="3"
BOOST_TOP_MS="3"

for BOOST_STOP in $BOOST_STOPS; do
    for BOOST_TOP_M in $BOOST_TOP_MS; do
        if [[ $BOOST_STOP -eq 2 && $BOOST_TOP_M -eq 1 ]]; then
            continue
        fi
        ATTEMPT="sb-3257_2-fix-${BOOST_STOP}-${BOOST_TOP_M}"
        EXP_BASE_NAME=$ATTEMPT
        mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}

        log_file="logs/bench-${ATTEMPT}-$(date +%Y%m%d%H%M%S).log"

        mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT
        mkdir -p $RESULT_DIR/${EXP_NAME}_ppl
        mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,stor}/mxq/$ATTEMPT

        # MODELS="0"
        EXP_NAME="${ATTEMPT}"
        echo "=========Run perplexity evaluation========="
        python ../src/cli.py llm \
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
        echo "=========Collect perplexity evaluation result on batch ${EXP_NAME}========="
        find $RESULT_DIR/${EXP_NAME}_ppl \
          -name "result-*.csv" \
          -printf '%T@ %p\n' \
          | sort -n \
          | tail -1 \
          | cut -d' '  -f2 \
          | xargs -i cp {} $RESULT_DIR/$EXP_BASE_NAME/data/ppl/mxq/$ATTEMPT

        echo "=========Dump quantization configs on batch ${EXP_NAME}========="
        mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT
        python ../src/cli.py dump \
          --type quant_config \
          --model $MODELS \
          --budget ${BUDGETS} \
          --attempt $ATTEMPT \
          --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
          --output-file "$RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT/quant-allot-${EXP_NAME}.csv" \
          2>&1 \
          | tee -a $log_file

        echo "=========Run memory evaluation on batch ${EXP_NAME}========="
        algo=mxq
        model_ids=$MODELS
        for m in $model_ids; do
          for cfg in ${BUDGETS}; do
              python ../src/cli.py llm \
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
        echo "=========Collect memory evaluation result on batch ${EXP_NAME}========="
        find $RESULT_DIR/${EXP_NAME}_stor \
          -name "result-*.csv" \
          -printf '%T@ %p\n' \
          | sort -n \
          | tail -1 \
          | cut -d' '  -f2 \
          | xargs -i cp {} $RESULT_DIR/$EXP_BASE_NAME/data/stor/mxq/$ATTEMPT

        # echo "=========Delete quantized models of batch ${batch_name}========="
        # find $QUANT_SNAPSHOT_DIR/$ATTEMPT -maxdepth 1 -type d | xargs rm -fr

        OLD_DIR=$(pwd)
        cd $RESULT_DIR/$EXP_BASE_NAME
        if [ ! -d pdfs/allot ]; then
            mkdir -p pdfs/allot
        fi
        $OLD_DIR/../data-vis/combine.R \
            --baseline_data_dir $OLD_DIR/../data-vis/data \
            --mxq_data_dir data

        cd $OLD_DIR
    done
done


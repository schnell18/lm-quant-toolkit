TOOLKIT_DIR="../../.." \
RESULT_BASE_DIR="/fdata/llm/mxq/results"


ATTMPTS="
kurt-ab-2-0
kurt-ab-2-1
kurt-ab-2-2
kurt-ab-2-3
kurt-ab-3-0
kurt-ab-3-1
kurt-ab-3-2
kurt-ab-3-3
kurt-boost-2-0
kurt-boost-2-1
kurt-boost-2-2
kurt-boost-2-3
kurt-boost-3-0
kurt-boost-3-1
kurt-boost-3-2
kurt-boost-3-3
sensi-boost-2-0
sensi-boost-2-1
sensi-boost-2-2
sensi-boost-2-3
sensi-boost-3-0
sensi-boost-3-1
sensi-boost-3-2
sensi-boost-3-3
sensi-ablation-2-1
sensi-ablation-2-2
"

# copy PPL results
mkdir -p "data/ppl/mxq"
mkdir -p "data/stor/mxq"
mkdir -p "data/allot/mxq"
for attempt in $ATTMPTS; do
  cp -r "$RESULT_BASE_DIR/$attempt/data/ppl/mxq/$attempt" data/ppl/mxq
  cp -r "$RESULT_BASE_DIR/$attempt/data/stor/mxq/$attempt" data/stor/mxq
  cp -r "$RESULT_BASE_DIR/$attempt/data/allot/mxq/$attempt" data/allot/mxq
done

$TOOLKIT_DIR/data-vis/combine.R \
    --baseline_data_dir $TOOLKIT_DIR/data-vis/data \
    --mxq_data_dir data


TOOLKIT_DIR="../../.." \
RESULT_BASE_DIR="/fdata/llm/mxq/results"


ATTMPTS="
sensi-milp-1
sensi-milp-2
sensi-milp-3
kurt-milp-1
kurt-milp-2
kurt-milp-3
sensi-milp-abl-1
kurt-milp-abl
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

# copy mxq2 results
cp -r "$TOOLKIT_DIR/data-vis/data/ppl/mxq/mxq2" data/ppl/mxq
cp -r "$TOOLKIT_DIR/data-vis/data/stor/mxq/mxq2" data/stor/mxq
cp -r "$TOOLKIT_DIR/data-vis/data/allot/mxq/mxq2" data/allot/mxq

# rename sensi-milp-abl-1 to sensi-milp-abl
mv data/ppl/mxq/sensi-milp-abl-1 data/ppl/mxq/sensi-milp-abl
mv data/stor/mxq/sensi-milp-abl-1 data/stor/mxq/sensi-milp-abl
mv data/allot/mxq/sensi-milp-abl-1 data/allot/mxq/sensi-milp-abl

$TOOLKIT_DIR/data-vis/combine.R \
    --baseline_data_dir $TOOLKIT_DIR/data-vis/data \
    --mxq_data_dir data


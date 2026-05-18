TOOLKIT_DIR="../../../.." \
RESULT_BASE_DIR="/fdata/llm/ieee-tai/results4"


ATTMPTS="
kurt-boost-2-0
kurt-boost-2-1
kurt-boost-2-2
kurt-boost-2-3
kurt-boost-3-0
kurt-boost-3-1
kurt-boost-3-2
kurt-boost-3-3
"

for attempt in $ATTMPTS; do
  mkdir -p "data/qnt/mxq/${attempt}"
  cp $RESULT_BASE_DIR/${attempt}_qnt/result*.csv data/qnt/mxq/${attempt}

  mkdir -p "data/ppl/mxq/${attempt}"
  cp $RESULT_BASE_DIR/${attempt}_ppl/result-*.csv data/ppl/mxq/${attempt}

  mkdir -p "data/allot/mxq/${attempt}"
  cp $RESULT_BASE_DIR/${attempt}_cfg/*.csv data/allot/mxq/${attempt}

  # copy the last results*.csv file to reduce duplication
  stor_csv=$(ls $RESULT_BASE_DIR/${attempt}_stor/result*.csv | xargs wc -l | grep result- | sort -k 1 -n | tail -1 | sed -e 's/^\s\+//' | cut -d' ' -f2)
  mkdir -p "data/stor/mxq/${attempt}"
  cp $stor_csv data/stor/mxq/${attempt}
done

# $TOOLKIT_DIR/data-vis/combine.R \
#     --baseline_data_dir $TOOLKIT_DIR/data-vis/ieee-tai/data \
#     --mxq_data_dir data

#!/bin/bash

source as_0_set_path.sh
pushd $CURRENTDIR/../

#----------------------------------
# Auto-split-v1 results accumulate
#----------------------------------
function collect_accuracy_v1 {
  for val in "${!Timestamp_v1[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v1[$val]}
    echo "Collect Accuracy auto-split v1: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    $PYTHON tools/run_quantization/get_accuracy.py -n $TIMESTAMPFOLDER
  done
}

#----------------------------------
# Auto-split-v2 results accumulate
#----------------------------------
function collect_accuracy_v2 {
  for val in "${!Timestamp_v2[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v2[$val]}
    echo "Collect Accuracy auto-split v2: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    $PYTHON tools/run_quantization/get_accuracy.py -n $TIMESTAMPFOLDER
  done
}

### Step 2.3 Collect accuracy stats from the previous run
collect_accuracy_v1
#collect_accuracy_v2
popd




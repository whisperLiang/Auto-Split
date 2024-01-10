#!/bin/bash

source as_0_set_path.sh
pushd $CURRENTDIR/../


function gen_latency_v1 {
  for val in "${!Timestamp_v1[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v1[$val]}
    echo "Generate Latency auto-split v1: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    $PYTHON tools/bit_search/model_latency_2_gen.py -n $TIMESTAMPFOLDER -t 16
#    -m $MINMSE
  done
}

function gen_latency_v2 {
  for val in "${!Timestamp_v2[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v2[$val]}
    echo "Generate Latency auto-split v2: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    $PYTHON tools/bit_search/model_latency_2_gen.py -n $TIMESTAMPFOLDER -t 16 -m $MINMSE
  done
}


echo 'Running Stats from Auto-split-v1'
gen_latency_v1
#echo 'Running Stats from Auto-split-v2'
#gen_latency_v2

popd

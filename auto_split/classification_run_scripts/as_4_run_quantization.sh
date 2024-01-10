#!/bin/bash

source as_0_set_path.sh
pushd $CURRENTDIR/../

#----------------------------------
# Auto-split-v1 results accumulate
#----------------------------------

function collect_latency_v1 {
  for val in "${!Timestamp_v1[@]}"; do
    echo "collect latency auto-split v1: ${val} - ${Timestamp_v1[$val]}"
    export TIMESTAMPFOLDER=${Timestamp_v1[$val]}
    $PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
  done
}

function set_yaml_v1 {
  for val in "${!Timestamp_v1[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v1[$val]}
    echo "set yaml auto-split v1: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    $PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
  done
}

function run_quantization_v1 {
  for val in "${!Timestamp_v1[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v1[$val]}
    echo "Run quantization auto-split v1: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    DEVICEID=$1
    CUDA_VISIBLE_DEVICES=$DEVICEID  $PYTHON  tools/run_quantization/run_distiller.py -n $TIMESTAMPFOLDER --arch $MODELNAME $DATASET --deviceid $DEVICEID --pythonexec $PYTHON
  done
}

function collect_accuracy_v1 {
  for val in "${!Timestamp_v1[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v1[$val]}
    echo "Collect Accuracy auto-split v1: ${MODELNAME} - ${TIMESTAMPFOLDER}"
#    $PYTHON tools/run_quantization/get_accuracy.py -n $TIMESTAMPFOLDER
  done
}

#----------------------------------
# Auto-split-v2 results accumulate
#----------------------------------


function collect_latency_v2 {
  for val in "${!Timestamp_v2[@]}"; do
    echo "collect latency auto-split v2: ${val} - ${Timestamp_v2[$val]}"
    export TIMESTAMPFOLDER=${Timestamp_v2[$val]}
    $PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
  done
}

function set_yaml_v2 {
  for val in "${!Timestamp_v2[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v2[$val]}
    echo "set yaml auto-split v2: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    $PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
  done
}

function run_quantization_v2 {
  for val in "${!Timestamp_v2[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v2[$val]}
    echo "Run quantization auto-split v2: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    DEVICEID=$1
    CUDA_VISIBLE_DEVICES=$DEVICEID  $PYTHON  tools/run_quantization/run_distiller.py -n $TIMESTAMPFOLDER --arch $MODELNAME $DATASET --deviceid $DEVICEID --pythonexec $PYTHON
  done
}

function collect_accuracy_v2 {
  for val in "${!Timestamp_v2[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v2[$val]}
    echo "Collect Accuracy auto-split v2: ${MODELNAME} - ${TIMESTAMPFOLDER}"
#    $PYTHON tools/run_quantization/get_accuracy.py -n $TIMESTAMPFOLDER
  done
}

#$PYTHON tools/run_quantization/run_distiller_yolo.py --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile

### Collect latency stats from previous runs of auto-split
collect_latency_v1
#collect_latency_v2
##
### Step 2.1: Set yaml configurations for all bit configurations
set_yaml_v1
#set_yaml_v2

### Step 2.2: Run various bit configurations on distiller LAPQ
#export CUDA_VISIBLE_DEVICES=3
DEVICEID=3
run_quantization_v1 $DEVICEID
#run_quantization_v2

### Step 2.3 Collect accuracy stats from the previous run
#collect_accuracy_v1
#collect_accuracy_v2

popd


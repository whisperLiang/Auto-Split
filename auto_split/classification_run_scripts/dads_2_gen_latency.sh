#!/bin/bash



CURRENTDIR=$PWD
pushd $CURRENTDIR/../

export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master

#.31
#export PYTHON=~/anaconda3/envs/pyt_13/bin/python

# .117
export PYTHON=~/anaconda3/envs/env1/bin/python


#APPS=('yolov3-spp-512' 'yolov3-spp-416' 'yolov3-spp-608' 'yolov3-416' 'yolov3-512' 'yolov3-608' 'yolov3-tiny-608' 'yolov3-tiny-416' 'yolov3-tiny-512')
#declare -A  Timestamp_v1=(['resnext50_32x4d']='resnext50_32x4d_20200927-134451' \
#  ['googlenet']='googlenet_20200927-134451' ['resnet50']='resnet50_20200927-134451' \
#  ['resnet18']='resnet18_20200927-134450' ['mobilenet_v2']='mobilenet_v2_20200927-134452' \
#  ['mnasnet1_0']='mnasnet1_0_20200927-134452')

#  yolov3-416_20200927-193015  yolov3-spp-416_20200927-193014  yolov3-tiny-416_20200927-193018
#yolov3-512_20200927-193016  yolov3-spp-512_20200927-193013  yolov3-tiny-512_20200927-193018
#yolov3-608_20200927-193016  yolov3-spp-608_20200927-193014  yolov3-tiny-608_20200927-193017
#APPS=('yolov3-spp-512'  'yolov3-spp-608' 'yolov3-416' 'yolov3-512' 'yolov3-608' 'yolov3-tiny-608' 'yolov3-tiny-416' 'yolov3-tiny-512')
#declare -A Timestamp_v1=(['yolov3-spp-416']='yolov3-spp-416_20200927-193014' \
#['yolov3-tiny-416-v1']='yolov3-tiny-416_20200927-193018' ['yolov3-416']='yolov3-416_20200927-193015')

# W16A16 -- (['resnet50']='resnet50_20201005-180032')
# W8A8 -- resnet50_20201005-183139
#declare -A Timestamp_v1=(['resnet50']='resnet50_20201005-180032')
declare -A Timestamp_v1=(['resnet50']='resnet50_20201005-183139')
function gen_latency_v1 {
  for val in "${!Timestamp_v1[@]}"; do
    MODELNAME=${val}
    export TIMESTAMPFOLDER=${Timestamp_v1[$val]}
    echo "Generate Latency auto-split v1: ${MODELNAME} - ${TIMESTAMPFOLDER}"
    $PYTHON tools/bit_search/dads_latency_2_gen.py -n $TIMESTAMPFOLDER -t 8
  done
}


echo 'Running Stats from Auto-split-v1'
gen_latency_v1


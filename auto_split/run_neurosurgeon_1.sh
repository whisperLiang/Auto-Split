#!/bin/bash

APPS=('resnet18' \
      'resnet50' \
      'googlenet' \
      'mobilenet_v2' \
      'resnext50_32x4d' \
      'mnasnet1_0' 'yolov3-spp-416' 'yolov3-416' 'yolov3-tiny-416')

#APPS=('mnasnet1_0' 'mobilenet_v2')
#APPS=('yolov3-spp-416' 'yolov3-416' 'yolov3-tiny-416')

export PYTHONPATH=~/edge-cloud-collaboration/auto_split
export PYTHON=~/anaconda3/envs/env1/bin/python

for i in ${APPS[@]};do
  echo $i
  $PYTHON tools/bit_search/neurosurgeon_1_gen.py --arch $i
done
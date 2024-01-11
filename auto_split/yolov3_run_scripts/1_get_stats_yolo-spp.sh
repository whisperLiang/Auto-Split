#!/bin/bash

export  ROOTFOLDER=~/edge-cloud-collaboration/auto_split
pushd $ROOTFOLDER

export PYTHON=~/anaconda3/envs/pyt_13/bin/python
#export PYTHON=~/anaconda3/envs/env1/bin/python
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
export CUDA_VISIBLE_DEVICES=0

export yolocfg=~/edge-cloud-collaboration/auto_split/detection_models/yolov3_master/cfg/yolov3-spp.cfg
export wgtsfile=<path to >/yolov3_data/yolov3-spp.pt
#export cocodir=/data/<path to >/datasets/coco2017

##FOR IMAGE-608
#export IMGSIZE=608
#export MODELNAME=yolov3-spp-608
#CUDA_VISIBLE_DEVICES=0 $PYTHON detection_models/yolov3_master/quantize_edge_dnn.py -b 16 -j 16 --img-size $IMGSIZE --cfg $yolocfg --weights $wgtsfile  --evaluate --quantize-eval --qe-lapq

#FOR IMAGE-416
export IMGSIZE=416
export MODELNAME=yolov3-spp-416
CUDA_VISIBLE_DEVICES=0 $PYTHON detection_models/yolov3_master/quantize_edge_dnn.py -b 16 -j 16 --img-size $IMGSIZE --cfg $yolocfg --weights $wgtsfile  --evaluate --quantize-eval --qe-lapq


#FOR IMAGE-512
export IMGSIZE=512
export MODELNAME=yolov3-spp-512
CUDA_VISIBLE_DEVICES=0 $PYTHON detection_models/yolov3_master/quantize_edge_dnn.py -b 16 -j 16 --img-size $IMGSIZE --cfg $yolocfg --weights $wgtsfile  --evaluate --quantize-eval --qe-lapq

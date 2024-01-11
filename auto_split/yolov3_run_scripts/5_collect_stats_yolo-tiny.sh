#!/bin/bash

export  ROOTFOLDER=~/edge-cloud-collaboration/auto_split
pushd $ROOTFOLDER

#export PYTHON=~/anaconda3/envs/pyt_13/bin/python
export PYTHON=~/anaconda3/envs/env1/bin/python
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
export CUDA_VISIBLE_DEVICES=2

export yolocfg=~/edge-cloud-collaboration/auto_split/detection_models/yolov3_master/cfg/yolov3-tiny.cfg
export wgtsfile=<path to >/yolov3_data/yolov3-tiny.pt
#export cocodir=/data/<path to >/datasets/coco2017

#-------------------------------------
#-------------------------------------
#FOR IMAGE-416
export IMGSIZE=416
export MODELNAME=yolov3-tiny-416
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-tiny-416_20200910-033828
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-tiny-416_20200910-033836
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER


#-------------------------------------
#-------------------------------------
#FOR IMAGE-608
export IMGSIZE=608
export MODELNAME=yolov3-tiny-608
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-tiny-608_20200910-033904
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-tiny-608_20200910-033913
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER

#-------------------------------------
#-------------------------------------
#FOR IMAGE-512
export IMGSIZE=512
export MODELNAME=yolov3-tiny-512
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-tiny-512_20200910-033844
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-tiny-512_20200910-033853
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER



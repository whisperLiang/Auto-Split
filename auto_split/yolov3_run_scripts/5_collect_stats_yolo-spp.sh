#!/bin/bash

export  ROOTFOLDER=~/edge-cloud-collaboration/auto_split
pushd $ROOTFOLDER

#export PYTHON=~/anaconda3/envs/pyt_13/bin/python
export PYTHON=~/anaconda3/envs/env1/bin/python
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
export CUDA_VISIBLE_DEVICES=1

export yolocfg=~/edge-cloud-collaboration/auto_split/detection_models/yolov3_master/cfg/yolov3-spp.cfg
export wgtsfile=<path to>/yolov3-spp.pt

#-------------------------------------
#-------------------------------------
#FOR IMAGE-416
export IMGSIZE=416
export MODELNAME=yolov3-spp-416
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-spp-416_20200910-033838
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-spp-416_20200910-033912
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER


#-------------------------------------
#-------------------------------------
#FOR IMAGE-608
export IMGSIZE=608
export MODELNAME=yolov3-spp-608
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-spp-608_20200910-034344
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-spp-608_20200910-034428
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER

#-------------------------------------
#-------------------------------------
#FOR IMAGE-512
export IMGSIZE=512
export MODELNAME=yolov3-spp-512
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-spp-512_20200910-034105
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-spp-512_20200910-034144
#-------------------------------------
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER



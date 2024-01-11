#!/bin/bash

export  ROOTFOLDER=~/edge-cloud-collaboration/auto_split
pushd $ROOTFOLDER

#export PYTHON=~/anaconda3/envs/pyt_13/bin/python
export PYTHON=~/anaconda3/envs/env1/bin/python
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
export CUDA_VISIBLE_DEVICES=4

export yolocfg=~/edge-cloud-collaboration/auto_split/detection_models/yolov3_master/cfg/yolov3.cfg
export wgtsfile=<path to >/yolov3_data/yolov3.pt
#export cocodir=/data/<path to >/datasets/coco2017

#-------------------------------------
#-------------------------------------
#FOR IMAGE-416
export IMGSIZE=416
export MODELNAME=yolov3-416
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-416_20200910-033845
#-------------------------------------
# Collect Latency Stats
$PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
# set yaml config
$PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
#3: Run distiller
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON tools/run_quantization/run_distiller_yolo.py --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-416_20200910-033934
#-------------------------------------
# Collect Latency Stats
$PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
# set yaml config
$PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
#3: Run distiller
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON tools/run_quantization/run_distiller_yolo.py --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER


#-------------------------------------
#-------------------------------------
#FOR IMAGE-608
export IMGSIZE=608
export MODELNAME=yolov3-608
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-608_20200910-034447
#-------------------------------------
# Collect Latency Stats
$PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
# set yaml config
$PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
#3: Run distiller
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON tools/run_quantization/run_distiller_yolo.py --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-608_20200910-034546
#-------------------------------------
# Collect Latency Stats
$PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
# set yaml config
$PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
#3: Run distiller
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON tools/run_quantization/run_distiller_yolo.py --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER

#-------------------------------------
#-------------------------------------
#FOR IMAGE-512
export IMGSIZE=512
export MODELNAME=yolov3-512
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-512_20200910-034140
#-------------------------------------
# Collect Latency Stats
$PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
# set yaml config
$PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
#3: Run distiller
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON tools/run_quantization/run_distiller_yolo.py --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER
#-------------------------------------
# Auto split -v2
export TIMESTAMPFOLDER=yolov3-512_20200910-034233
#-------------------------------------
# Collect Latency Stats
$PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
# set yaml config
$PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
#3: Run distiller
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON tools/run_quantization/run_distiller_yolo.py --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER



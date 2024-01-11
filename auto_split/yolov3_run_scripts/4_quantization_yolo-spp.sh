#!/bin/bash

export  ROOTFOLDER=~/edge-cloud-collaboration/auto_split
pushd $ROOTFOLDER

#export PYTHON=~/anaconda3/envs/pyt_13/bin/python
export PYTHON=~/anaconda3/envs/env1/bin/python
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
export CUDA_VISIBLE_DEVICES=1

export yolocfg=~/edge-cloud-collaboration/auto_split/detection_models/yolov3_master/cfg/yolov3-spp.cfg
export wgtsfile=<path to >/yolov3_data/yolov3-spp.pt
#export cocodir=/data/<path to >/datasets/coco2017

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
export TIMESTAMPFOLDER=yolov3-spp-416_20200910-033912
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
export MODELNAME=yolov3-spp-608
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-spp-608_20200910-034344
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
export TIMESTAMPFOLDER=yolov3-spp-608_20200910-034428
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
export MODELNAME=yolov3-spp-512
#-------------------------------------
#-------------------------------------
# Auto split -v1
export TIMESTAMPFOLDER=yolov3-spp-512_20200910-034105
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
export TIMESTAMPFOLDER=yolov3-spp-512_20200910-034144
#-------------------------------------
# Collect Latency Stats
$PYTHON tools/bit_search/model_latency_3_collect.py -n $TIMESTAMPFOLDER
# set yaml config
$PYTHON tools/run_quantization/set_yaml_config_lapq.py -n $TIMESTAMPFOLDER --arch $MODELNAME
#3: Run distiller
CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES $PYTHON tools/run_quantization/run_distiller_yolo.py --pythonpath $PYTHON --img-size $IMGSIZE --deviceid $CUDA_VISIBLE_DEVICES --arch $MODELNAME -n $TIMESTAMPFOLDER --cfg $yolocfg --weights $wgtsfile
##4: Collect stats
$PYTHON tools/run_quantization/get_accuracy_yolo.py -n $TIMESTAMPFOLDER



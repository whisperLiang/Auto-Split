#!/bin/bash

export  ROOTFOLDER=~/edge-cloud-collaboration/auto_split
pushd $ROOTFOLDER

#export PYTHON=~/anaconda3/envs/pyt_13/bin/python
export PYTHON=~/anaconda3/envs/env1/bin/python
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
export CUDA_VISIBLE_DEVICES=0

export yolocfg=~/edge-cloud-collaboration/auto_split/detection_models/yolov3_master/cfg/yolov3-spp.cfg
export wgtsfile=<path to >/yolov3_data/yolov3-spp.pt
#export cocodir=/data/<path to >/datasets/coco2017


#FOR IMAGE-416
export IMGSIZE=416
export MODELNAME=yolov3-spp-416
#export TIMESTAMPFOLDER=yolov3-spp-416_20200909-141731

# Add rank stats
$PYTHON tools/bit_search/add_rank_min_max_stats.py --arch $MODELNAME

# Auto-split v1
$PYTHON tools/bit_search/auto_split.py --arch $MODELNAME

# Auto-split v2
$PYTHON tools/bit_search/auto_split.py --arch $MODELNAME --logdir data/$MODELNAME


#FOR IMAGE-512
export IMGSIZE=512
export MODELNAME=yolov3-spp-512
#export TIMESTAMPFOLDER=yolov3-spp-416_20200909-141731

# Add rank stats
$PYTHON tools/bit_search/add_rank_min_max_stats.py --arch $MODELNAME

# Auto-split v1
$PYTHON tools/bit_search/auto_split.py --arch $MODELNAME

# Auto-split v2
$PYTHON tools/bit_search/auto_split.py --arch $MODELNAME --logdir data/$MODELNAME


#FOR IMAGE-512
export IMGSIZE=608
export MODELNAME=yolov3-spp-608
#export TIMESTAMPFOLDER=yolov3-spp-416_20200909-141731

# Add rank stats
$PYTHON tools/bit_search/add_rank_min_max_stats.py --arch $MODELNAME

# Auto-split v1
$PYTHON tools/bit_search/auto_split.py --arch $MODELNAME

# Auto-split v2
$PYTHON tools/bit_search/auto_split.py --arch $MODELNAME --logdir data/$MODELNAME
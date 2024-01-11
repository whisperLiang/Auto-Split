#!/bin/bash

export PYTHON=~/anaconda3/envs/pyt_13/bin/python
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
export CUDA_VISIBLE_DEVICES=0
export yolocfg=~/edge-cloud-collaboration/auto_split/detection_models/yolov3_master/cfg/yolov3.cfg
export wgtsfile=<path to >/yolov3_data/yolov3.pt
export IMGSIZE=416
export MODELNAME=yolov3-416

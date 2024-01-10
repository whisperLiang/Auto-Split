#!/bin/bash

CURRENTDIR=$PWD
ROOTFOLDER=~/distributed-inference/auto_split
export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master
GETSTATSYAML=$CURRENTDIR/get_stats_pq.yaml
export PYTHON=~/anaconda3/envs/env1/bin/python
#export DATASET=/data/<path to >/datasets/ImageNet2017/ILSVRC/Data/CLS-LOC

#APPS=('googlenet' 'mnasnet1_0' 'mobilenet_v2' 'resnet50' 'resnet18' 'resnext50_32x4d')
#APPS=('googlenet' 'mnasnet1_0' 'mobilenet_v2' 'resnext50_32x4d')
#APPS=('resnet18')
#declare -A Timestamp_v2=(['resnet18']='resnet18_20200923-160827')
#declare -A Timestamp_v1=(['resnet18']='resnet18_20210205-203940')
#declare -A Timestamp_v1=(['resnet50']='resnet50_20210205-204000')
#declare -A  Timestamp_v1=(['googlenet']='googlenet_20210207-145458' ['mnasnet1_0']='mnasnet1_0_20210207-145824' ['mobilenet_v2']='mobilenet_v2_20210207-145912' ['resnext50_32x4d']='resnext50_32x4d_20210207-150121')
#APPS=('googlenet')
#declare -A  Timestamp_v1=(['googlenet']='googlenet_20210207-145458')
#APPS=('mnasnet1_0')
#declare -A  Timestamp_v1=(['mnasnet1_0']='mnasnet1_0_20210207-145824')
#APPS=('mobilenet_v2')
#declare -A  Timestamp_v1=(['mobilenet_v2']='mobilenet_v2_20210207-145912')

APPS=('resnext50_32x4d')
declare -A  Timestamp_v1=(['resnext50_32x4d']='resnext50_32x4d_20210207-150121')



#!/bin/bash




APPS=('googlenet' 'inception_v3' 'mnasnet1_0' 'mobilenet_v2' 'resnet18' 'resnet50' 'resnext50_32x4d' 'vgg16')
#APPS=('googlenet' 'mnasnet1_0' 'mobilenet_v2' 'resnet18' 'resnet50' 'resnext50_32x4d')
APPS=('googlenet' 'mnasnet1_0' 'mobilenet_v2' 'resnet50' 'resnext50_32x4d')
#APPS=('resnet18')
#APPS=('resnet18' 'resnet50' 'resnext50_32x4d' 'googlenet' 'mnasnet1_0' 'mobilenet_v2' )
APPS=('yolov3-spp-512' 'yolov3-spp-416' 'yolov3-spp-608' 'yolov3-416' 'yolov3-512' 'yolov3-608' 'yolov3-tiny-608' 'yolov3-tiny-416' 'yolov3-tiny-512')
CURRENTDIR=$PWD
pushd $CURRENTDIR/../

export PYTHONPATH=~/edge-cloud-collaboration/auto_split:~/edge-cloud-collaboration/libs/distiller-master

#.31
#export PYTHON=~/anaconda3/envs/pyt_13/bin/python

# .117
export PYTHON=~/anaconda3/envs/env1/bin/python



#function set_post_train_yaml {
#
#  for val in ${APPS[@]}; do
#     echo $val
#     export MODELNAME=$val
#     #1. Generate post_train.yaml file
#     post_train_dir=${CURRENTDIR}/post_train_yaml
#     mkdir -p $post_train_dir
#     sed  's/MODELNAME/'${MODELNAME}'/' ${CURRENTDIR}/sample_post_train.yaml > ${post_train_dir}/${MODELNAME}_post_train.yaml
#  done
#}
#
#function add_rank_stats {
#  # 2. Add min/max rank
#  for val in ${APPS[@]}; do
#     echo $val
#     export MODELNAME=$val
#     $PYTHON tools/bit_search/add_rank_min_max_stats.py --arch $MODELNAME
#  done
#}

function dads_v1 {
  for val in ${APPS[@]}; do
     echo $val
     export MODELNAME=$val
     $PYTHON tools/bit_search/dads.py --arch $MODELNAME
  done
}

### 1. Create post_train.yaml scripts for the benchmarks
#set_post_train_yaml
### 2. Add min/max rank
#add_rank_stats
### Run auto-split-v1
dads_v1

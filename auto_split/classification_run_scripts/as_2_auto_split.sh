#!/bin/bash

source as_0_set_path.sh
pushd ${CURRENTDIR}/../

function set_post_train_yaml {

  for val in ${APPS[@]}; do
     echo $val
     export MODELNAME=$val
     #1. Generate post_train.yaml file
     post_train_dir=${CURRENTDIR}/post_train_yaml
     mkdir -p $post_train_dir
     sed  's/MODELNAME/'${MODELNAME}'/' ${CURRENTDIR}/sample_post_train.yaml > ${post_train_dir}/${MODELNAME}_post_train.yaml
  done
}

function add_rank_stats {
  # 2. Add min/max rank
  for val in ${APPS[@]}; do
     echo $val
     export MODELNAME=$val
     $PYTHON tools/bit_search/add_rank_min_max_stats.py --arch $MODELNAME
  done
}

function auto_split_v1 {
  for val in ${APPS[@]}; do
     echo $val
     export MODELNAME=$val
     $PYTHON tools/bit_search/auto_split.py --arch $MODELNAME
  done
}

function auto_split_v2 {
  for val in ${APPS[@]}; do
     echo $val
     export MODELNAME=$val
     $PYTHON tools/bit_search/auto_split.py --arch $MODELNAME --logdir data/$MODELNAME
  done
}

## 1. Create post_train.yaml scripts for the benchmarks
set_post_train_yaml
## 2. Add min/max rank
add_rank_stats
# Run auto-split-v1
auto_split_v1
# Run auto-split-v2
#auto_split_v2


popd


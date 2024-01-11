#!/bin/bash

#declare -a VIDEOS=("face4.mp4" "face5.mp4" "face6.mp4" "face7.mp4" "face8.mp4" "face3.mp4"  )
#declare -a VIDEOS=("face8.mp4" )
PYTHON=~/anaconda3/envs/tf/bin/python
video_dir=~/distributed-inference/test_video


run_demo () {
  video=$1
  echo 'Post processing: ' $video
  cd cloud_dnn
  $PYTHON run_cloud.py --split-inference-video $video --gpu 0 --port 8000 &
  CloudTwoID=`echo $!`
  $PYTHON run_cloud.py --cloud-only-video  $video --gpu 1 --port 6000 &
  CloudOneID=`echo $!`

  cd ..
  sleep 15

  cd edge_dnn
  $PYTHON run_edge.py --split-inference-video $video -q --num-bits 4 --act-compress --ipaddr localhost --port 8000 &
  EdgeTwoID=`echo $!`
  $PYTHON run_edge.py --split-inference-video $video --cloud-only --ipaddr localhost --port 6000 &
  EdgeOneID=`echo $!`
  cd  ..

  wait
  echo 'Edge 1: PID: ' $EdgeOneID
  echo 'Edge 2: PID: ' $EdgeTwoID
  echo 'Cloud 1: PID: ' $CloudOneID
  echo 'Cloud 2: PID: ' $CloudTwoID
  echo 'All background processes have exited. For ' ${video}

}

# Iterate the string array using for loop
for i in {6..6}; do
  val='/f'${i}'.mp4'
  new_video=${video_dir}${val}
  echo $val
  run_demo $new_video
done

## Iterate the string array using for loop
#for val in ${VIDEOS[@]}; do
#  new_video=${video_dir}${val}
#  run_demo $new_video
#done












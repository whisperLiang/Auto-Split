#!/bin/bash
PYTHON=~/anaconda3/envs/tf/bin/python
#declare -a VIDEOS=("face4.mp4" "face5.mp4" "face6.mp4" "face7.mp4" "face8.mp4" "face3.mp4"  )
#declare -a VIDEOS=("face8.mp4" )
# Iterate the string array using for loop
#for val in ${VIDEOS[@]}; do
#  video=$val
#  cd cloud_dnn
#  echo 'Post processing: ' $video
#  $PYTHON concat_videos.py --video-name $video
#  cd ..
#done

# Iterate the string array using for loop
for i in {6..6}; do
  val='f'${i}'.mp4'
  video=$val
  cd cloud_dnn
  echo 'Post processing: ' $video
  $PYTHON concat_videos.py --video-name $video &
  cd ..
done
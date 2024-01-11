#!/bin/bash

PYTHON=~/anaconda3/envs/pyt_13/bin/python
VIDEO=~/distributed-inference/test_video/f6.mp4
IPADDR=localhost

cd cloud_dnn
$PYTHON run_cloud.py --cloud-only-video $VIDEO --gpu 1 --port 6000 &
$PYTHON run_cloud.py --split-inference-video $VIDEO --gpu 2  --port 8134 &
cd ..


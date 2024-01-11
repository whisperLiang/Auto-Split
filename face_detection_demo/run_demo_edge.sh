#!/bin/bash


PYTHON=~/anaconda3/envs/pyt_13/bin/python
VIDEO=~/distributed-inference/test_video/f6.mp4
IPADDR=localhost

cd edge_dnn
$PYTHON run_edge.py --split-inference-video $VIDEO --cloud-only --ipaddr $IPADDR --port 6000 &
$PYTHON run_edge.py  --split-inference-video $VIDEO -q --num-bits 4 --act-compress  --ipaddr $IPADDR --port 8134 &
cd ..
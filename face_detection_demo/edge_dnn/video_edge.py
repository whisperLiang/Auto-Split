# -*- coding: utf-8 -*-

import io
import base64
import os
import sys
import cv2
import time
import copy
import math
import numpy as np
import codecs
import logging
import random
import tensorflow as tf
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import argparse
from edge_split_inference import YoloSplitInference
from jpeg_compression import YoloSplitInference as YoloJPEG
import pathlib

class VideoInference():
    def __init__(self, args):
        cap, yolo_infer, is_quantized, num_bits = self.init_video_inference(args)
        self.cap = cap
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.yolo_infer = yolo_infer
        self.is_quantized = is_quantized
        self.num_bits = num_bits
        self.is_cloud_only = args.cloud_only
        self.act_compress = args.act_compress

    def init_video_inference(self, args):
        if args.cloud_only_video:
            test_dir_name = args.cloud_only_video
        elif args.split_inference_video:
            test_dir_name = args.split_inference_video
        else:
            raise ValueError('No input video is given ')

        is_quantized = args.is_quantized
        num_bits = args.num_bits
        yolo_infer = YoloSplitInference(args.ipaddr, args.port)

        assert '.mp4' in test_dir_name or '.avi' in test_dir_name, 'Not a video file '
        cap = cv2.VideoCapture(test_dir_name)
        return cap, yolo_infer, is_quantized, num_bits

    def run_split_video(self):
        # USE self.total_frames to run the entire video. For face1.mp4=1615.
        total=self.total_frames
        # total=200
        for i in range(total):
            self.yolo_infer.run_many_video(self.cap, self.is_quantized, self.num_bits,
                                           self.is_cloud_only, self.act_compress)

    def __del__(self):
        # -- release --
        print('Edge side destructor called')
        self.cap.release()
        # out.release()
        cv2.destroyAllWindows()
        del self.yolo_infer
        return

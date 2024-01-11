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
import pathlib

from video_edge import VideoInference
from image_edge_rpc import split_save_frozen_pb,split_inference,test_jpeg_compression

def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    if args.split_inference_video:
        print('SPLIT-VIDEO: testing pb model for %s' % args.split_inference_video)
        video_infer = VideoInference(args)
        video_infer.run_split_video()
        return

    elif args.split_inference:
        print('SPLIT-IMAGE: testing pb model for %s' % args.split_inference)
        split_inference(args)

    elif args.jpeg:
        print('SPLIT-JPEG: args %s' % args.jpeg)
        test_jpeg_compression(args)

    elif args.split_save_frozen_pb:
        print('Saving frozen .pb files for Edge and cloud DNN')
        split_save_frozen_pb()
    else:
        raise ValueError('arg not found.')



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--split-inference', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')
    parser.add_argument('--jpeg', default=None, type=lambda s: s.lower(),
                        help='jpeg compression, and pass directory containing test images')

    parser.add_argument('--split-inference-video', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')

    parser.add_argument('--cloud-only-video', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')

    parser.add_argument('-q', '--is-quantized', action='store_true',
                        help="shows output")

    parser.add_argument('--num-bits', metavar='test_dir', default=None, type=int,
                        help='bit width for transmission activation')

    parser.add_argument('--split-save-frozen-pb', action='store_true',
                        help="saves frozen .pb files for edge and cloud DNN")

    parser.add_argument('--edge-inference', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')

    parser.add_argument('--edge-save-frozen-pb', action='store_true',
                        help="saves frozen .pb files for edge and cloud DNN")

    parser.add_argument('--ipaddr', default='localhost', type=lambda s: s.lower(),
                        help='Provide the ip address')
    parser.add_argument('--port', default='8134', type=int,
                        help='Provide port number')

    parser.add_argument('--act-compress', action='store_true')
    parser.add_argument('--cloud-only', action='store_true')

    args = parser.parse_args()
    main(args)

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
from cloud_split_inference import YoloSplitInference
from cloud_only_inference import Yolo
from xmlrpc.server import SimpleXMLRPCServer
import socket
import sys
import cv2
import pickle
import numpy as np
import struct
import matplotlib.pyplot as pyplot
import time
import pathlib
import logging

def cloud_only_inference_server(input_bytes, scale, zero_point, image_data_shape, is_quantized, image_file_name,
    input_shape, act_compress, act_bitwidth):

    yolo_infer = Yolo()
    yolo_infer.run_many_server(input_bytes, scale, zero_point, image_data_shape, is_quantized, image_file_name,
    input_shape, act_compress, act_bitwidth)


class VideoInference():
    def __init__(self, args, logger):
        self.result_dir = 'results'
        pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        self.logger = logger
        if args.cloud_only_video:
            self.is_cloud_only = True
        else:
            self.is_cloud_only = False
        cap, yolo_infer, out = self.init_video_inference(args)
        self.cap = cap
        self.out = out
        self.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        self.yolo_infer = yolo_infer




    def init_video_inference(self, args):
        if args.cloud_only_video:
            test_dir_name = args.cloud_only_video
            yolo_infer = Yolo(self.logger, test_dir_name, self.result_dir)
        elif args.all_cloud_video:
            test_dir_name = args.all_cloud_video
            yolo_infer = Yolo(self.logger, test_dir_name, self.result_dir)
        elif args.split_inference_video:
            test_dir_name = args.split_inference_video
            yolo_infer = YoloSplitInference(self.logger, test_dir_name, self.result_dir)
        else:
            raise ValueError('No input video is given ')

        assert '.mp4' in test_dir_name or '.avi' in test_dir_name, 'Not a video file '
        cap = cv2.VideoCapture(test_dir_name)

        # Get current width of frame
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fourcc = cv2.VideoWriter_fourcc('m','p','4','v')

        name_prefix=test_dir_name.split('.mp4')[0].split('/')[-1]

        if self.is_cloud_only:
            FILE_OUTPUT = os.path.join(self.result_dir,"cloud_only_{}.mp4".format(name_prefix))
        else:
            FILE_OUTPUT = os.path.join(self.result_dir,"auto_split_{}.mp4".format(name_prefix))

        orig_fps = cap.get(cv2.CAP_PROP_FPS)
        out = cv2.VideoWriter(FILE_OUTPUT, fourcc, orig_fps, (width, height))

        return cap, yolo_infer, out

    def run_split_video(self, input_bytes, scale, zero_point, image_data_shape, is_quantized,
                        image_file_name, input_shape, act_compress, num_bits):

        self.yolo_infer.run_many_video_server(self.cap, self.out, input_bytes, scale, zero_point,
                              image_data_shape, is_quantized,
                              image_file_name, input_shape, act_compress, num_bits)


    def __del__(self):
        # -- release --
        # self.logger('VideoInference cloud destructor called')
        self.cap.release()
        self.out.release()
        cv2.destroyAllWindows()
        # self.yolo_infer.cleanup()
        return

def run_server_rpc(ipaddr, port, fn, logger, STOPSERVER=100):
    server = SimpleXMLRPCServer((ipaddr, port), allow_none=True)
    logger.info("Listening on {}:{}...".format(ipaddr, port))
    server.register_function(fn, "split_inference_server")
    server.serve_forever()

def run_server_socket(ipaddr, port, fn, logger, STOPSERVER):
    def recv_split_act(msg):
        HEADERSIZE = 10

        # 1. read Other vars header
        other_vars_len = int(msg[:HEADERSIZE])
        msg = msg[HEADERSIZE:]
        # 2. read xx pickle header
        xx_pickle_len = int(msg[:HEADERSIZE])
        msg = msg[HEADERSIZE:]

        # 3. read other_vars
        msg_other_vars = msg[:other_vars_len]
        msg = msg[other_vars_len:]
        other_vars = struct.unpack('ddIIII?IIII?I', msg_other_vars)

        # 4. read xx data
        xx_pickle = msg[:xx_pickle_len]
        xx = pickle.loads(xx_pickle)
        # print(xx.shape)

        return xx, other_vars


    HOST = ipaddr
    PORT = port
    HEADERSIZE = 10
    BYTES = 4096
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    logger.info('Socket created')

    s.bind((HOST, PORT))
    logger.info('Socket bind complete')
    s.listen(10)
    logger.info('Socket now listening at {}:{}'.format(HOST,PORT))
    conn, addr = s.accept()


    MAXMSG=1
    recv_msg=0
    i_msg=0
    while recv_msg <= STOPSERVER:
        full_msg = b''
        new_msg = True
        msglen = None
        while recv_msg <= STOPSERVER:
            msg = conn.recv(BYTES)
            if msg == b'':
                # print('Finished: Currently receiving empty packets')
                continue

            full_msg += msg
            if new_msg:
                # print("new msg len:", full_msg[:2 * HEADERSIZE])
                other_vars_len = int(full_msg[0:HEADERSIZE])
                act_len = int(full_msg[HEADERSIZE:2 * HEADERSIZE])
                msglen = 2 * HEADERSIZE + other_vars_len + act_len
                new_msg = False
                # print(f"full message length: {msglen}")


            if len(full_msg) >= msglen:
                logger.debug("full msg recvd #times: {}".format(i_msg))
                current_msg = full_msg[:msglen]
                full_msg = full_msg[msglen:]
                xx, other_vars = recv_split_act(current_msg)
                (scale, zero_point, B, H, W, C, is_quantized, frame_count,
                 HI, WI, CI, act_compress, num_bits) = other_vars
                if B==0:
                    input_shape = (H,W,C)
                else:
                    input_shape = (B,H,W,C)


                send_msg = 'ACK:,frame_count:,{},xx_shape:,{}'.format(frame_count, xx.shape)
                conn.sendall(send_msg.encode('utf-8'))
                if i_msg == MAXMSG -1 :
                    logger.debug('###### Function called for frame number: {}'.format(frame_count))
                    fn(xx, scale, zero_point, (HI, WI, CI), is_quantized, frame_count, input_shape, act_compress,
                       num_bits)
                    i_msg = 0
                    recv_msg +=1
                else:
                    i_msg += 1

                new_msg = True
                msglen = None



def main(args):
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    # gpu_device_name = tf.config.experimental.list_physical_devices('XLA_GPU')[0].name
    # cpu_device_name = tf.config.experimental.list_physical_devices('XLA_CPU')[0].name
    run_server = run_server_rpc
    run_server = run_server_socket

    # Set Logger
    logger = logging.getLogger('CloudDNN')
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.StreamHandler())
    logfile = pathlib.Path("cloud.log")
    if os.path.isfile(logfile):
        logfile.unlink()


    fh = logging.FileHandler('cloud.log')
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)
    formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    # logger.debug('Debug Message')
    # logger.info('Info Message')
    # logger.warning('Warning')
    # logger.error('Error Occured')
    # logger.critical('Critical Error')

    # Stop server after XX messages
    # STOPSERVER=160

    if args.split_inference_video:
        print('SPLIT-VIDEO: testing pb model for %s' % args.split_inference_video)
        video_infer = VideoInference(args, logger)
        STOPSERVER=video_infer.total_frames - 1
        run_server(args.ipaddr, args.port, video_infer.run_split_video, logger, STOPSERVER=STOPSERVER)
        return

    elif args.cloud_only_video:
        print('CLOUD-ONLY-VIDEO: testing pb model for %s' % args.split_inference_video)
        video_infer = VideoInference(args, logger)
        STOPSERVER = video_infer.total_frames - 1
        run_server(args.ipaddr, args.port, video_infer.run_split_video, logger, STOPSERVER=STOPSERVER)
        return

    elif args.all_cloud_video:
        print('All-CLOUD-VIDEO: testing pb model for %s' % args.split_inference_video)
        video_infer = VideoInference(args, logger)
        video_infer.yolo_infer.run_incloud_many_video_server(video_infer.cap)
        return

    elif args.split_inference:
        print('SPLIT-IMAGES: testing pb model for %s' % args.split_inference)
        yolo_infer = YoloSplitInference(logger,'results')
        STOPSERVER = yolo_infer.total_frames - 1
        run_server(args.ipaddr, args.port, yolo_infer.run_many_server, logger, STOPSERVER=STOPSERVER)
        return

    elif args.cloud_only:
        print('CLOUD ONLY: inference for images, %s' % args.cloud_only)
        STOPSERVER = 160 - 1
        run_server(args.ipaddr, args.port,cloud_only_inference_server, logger, STOPSERVER=STOPSERVER)
        return


if __name__ == "__main__":
    LOG_FILENAME = 'example.log'
    logging.basicConfig(filename=LOG_FILENAME, level=logging.DEBUG)

    parser = argparse.ArgumentParser()
    parser.add_argument('--split-inference', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')

    parser.add_argument('--cloud-only', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')

    parser.add_argument('--split-inference-video', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')

    parser.add_argument('--cloud-only-video', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')

    parser.add_argument('--all-cloud-video', default=None, type=lambda s: s.lower(),
                        help='directory containing test images')

    parser.add_argument('--ipaddr', default='', type=lambda s: s.lower(),
                        help='Provide the ip address')
    parser.add_argument('--port', default='8134', type=int,
                        help='Provide port number')
    parser.add_argument('--gpu', default='0', type=lambda s: s.lower(),
                        help='Provide gpu devices -- 0,1,2,3')

    args = parser.parse_args()
    main(args)


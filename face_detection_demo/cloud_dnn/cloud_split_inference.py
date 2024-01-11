# -*- coding: utf-8 -*-
"""
ssd inference interface

x00358726
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import codecs
import logging
import cv2
import numpy as np
import math
import csv
import pickle

import tensorflow as tf #pylint: disable=import-error
from google.protobuf import text_format
from tensorflow.core.framework import graph_pb2
from tensorflow.python.framework import tensor_util
from tensorflow.python.client import timeline

from config_person import YoloConfig
from yolo3_split_model import CloudDNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from bitstruct import unpack
import base64
from linear_quantization import Quantize
import pathlib

class YoloSplitInference(object):
    def __init__(self, logger, test_dir_name, result_dir):
        config = tf.ConfigProto(allow_soft_placement=True)
        config.gpu_options.allow_growth = True
        self.yolo_config = YoloConfig()
        self.frame = 1
        self.start_time =0.0
        self.prev_tend = self.start_time
        self.c_transmission =0
        self.c_decompress =0
        self.cloud_dnntime =0
        self.logger = logger
        self.result_dir = result_dir
        pathlib.Path(self.result_dir).mkdir(parents=True, exist_ok=True)

        # Set file to store.. timestamp
        time_logger = logging.getLogger('Time')
        time_logger.setLevel(logging.INFO)
        name_prefix=test_dir_name.split('.mp4')[0].split('/')[-1]

        time_logger_filename = os.path.join(self.result_dir,"auto_split_timestamp_{}.csv".format(name_prefix))
        logfile = pathlib.Path(time_logger_filename)
        if os.path.isfile(logfile):
            logfile.unlink()

        time_fh = logging.FileHandler(time_logger_filename)
        time_fh.setLevel(logging.INFO)
        time_logger.addHandler(time_fh)
        time_logger.addHandler(time_fh)
        self.time_logger = time_logger

        # Run one test case.
        self.load_checkpoint(self.yolo_config)
        return

    def __del__(self):
        self.sess_cloud.close()
        # -- release --
        self.logger.info('YoloSplitInference destructor called')
        return

    def load_checkpoint(self, yolo_config):
        self.cloud_dnn = CloudDNN(False, yolo_config)
        self.sess_cloud = tf.Session(graph=self.cloud_dnn.graph)
        self.cloud_dnn.load_checkpoint(self.sess_cloud)
        return


    def run_many_server(self, xx_encoded, scale, zero_point, image_data_shape, is_quantized, image_file_name,
                        input_shape, act_compress, act_bitwidth):

        if is_quantized:
            input_dtype = int
            run_one_fn = self.run_one_quantized_server
        else:
            input_dtype = float
            run_one_fn = self.run_one_server

        if act_compress:
            len_symbols=1
            for i in input_shape:
                len_symbols *= i


            # # Method 1
            # # C1. decode utf-8 string to bytes
            # xx_bytes = base64.b64decode(xx_encoded)
            # # C2. decode bytes to array/list

            # Method 2
            xx_bytes = bytes.fromhex(xx_encoded)

            dtype = 'u{}'.format(act_bitwidth) * len_symbols
            xx_encoded = unpack(dtype, xx_bytes)


        recovered_input_q = np.array(xx_encoded)
        recovered_input_q.resize(input_shape)
        # recovered_input_q_int = np.array(input_q_int_list)
        # recovered_input_q = recovered_input_q_int.astype(float)
        # diff = (recovered_input_q - input_q).sum()
        #0-------
        dir_name = './test_images/'
        if dir_name != '':
            dir_basename = os.path.basename(dir_name)
            image_out = os.path.join('./result', dir_basename)
        else:
            image_out = './result'

        if not os.path.exists(image_out):
            os.mkdir(image_out)

        image_file = image_file_name
        self.logger.debug('image_file : ', image_file)
        img_data = cv2.imread(os.path.join(dir_name, image_file))
        image_dir_name = image_file.strip('.jpg')
        image_dir = os.path.join('..','output',image_dir_name)
        time_start = time.time()
        labels, scores, bboxes = run_one_fn(recovered_input_q, scale, zero_point, image_data_shape)
        total_time = time.time() - time_start
        self.logger.debug('{},{},{} '.format(image_file, img_data.shape,total_time))
        img_to_draw = self.bboxes_draw_on_img_cv(img_data, bboxes, scores, labels, self.frame,0)
        # plt.imshow(img_to_draw)
        # plt.show()
        cv2.imwrite(os.path.join(image_out, image_file), img_to_draw)
        final_image_path = os.path.join(image_out,image_file_name)
        plt.imsave(final_image_path, img_to_draw)
        # plt.show()
        return

    def run_many_video_server(self, cap, out, xx_encoded, scale, zero_point,
                                       image_data_shape, is_quantized,
                                       image_file_name, input_shape, act_compress, act_bitwidth):
        act_compress_2 = False
        time1 = time.time()
        if self.frame <=1:
            self.start_time = time1
            self.prev_tend = time1

        if is_quantized:
            run_one_fn = self.run_one_quantized_server
        else:
            run_one_fn = self.run_one_server

        if act_compress:
            def decompress(input_packed, H, W, C, C_by_2, num_bits):
                recovered_input = np.zeros([H, W, C], dtype=np.uint8)
                for i in range(0, C_by_2):
                    l_x = np.left_shift(input_packed[:, :, i], num_bits)
                    y = np.right_shift(l_x, num_bits)
                    x = np.right_shift(input_packed[:, :, i], num_bits)
                    recovered_input[:, :, 2 * i] = x
                    recovered_input[:, :, 2 * i + 1] = y

                return recovered_input

            (b,H,W,C) = input_shape
            C_by_2 = int(math.ceil(C / 2))
            input_packed = np.array(xx_encoded,dtype=np.uint8)
            input_packed.resize((H,W,C_by_2))
            recovered_input_q = np.zeros(input_shape, dtype=np.uint8)
            recovered_input_q[0] = decompress(input_packed, H, W, C, C_by_2, act_bitwidth)
            # recovered_input_q.resize(input_shape)
            recovered_input_q = recovered_input_q.astype(float)
        else:
            recovered_input_q = np.array(xx_encoded)
            recovered_input_q = recovered_input_q.astype(float)
            recovered_input_q.resize(input_shape)

        if act_compress_2:
            len_symbols=1
            for i in input_shape:
                len_symbols *= i

            # Method 2
            xx_bytes = bytes.fromhex(xx_encoded)

            dtype = 'u{}'.format(act_bitwidth) * len_symbols
            xx_encoded = unpack(dtype, xx_bytes)

        time2 = time.time()


        labels, scores, bboxes = run_one_fn(recovered_input_q, scale, zero_point, image_data_shape)

        try:
            cap.isOpened()
            ret, frame = cap.read()
            # img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            img_data = frame
            time3 = time.time()

            t_prednn_and_trans = time1 - self.prev_tend
            self.c_transmission += t_prednn_and_trans

            t_decompress = time2 - time1
            self.c_decompress += t_decompress

            t_cloud_dnntime = time3 - time2
            self.cloud_dnntime += t_cloud_dnntime

            avg_prednn_and_trans = self.c_transmission/self.frame
            avg_decompress = self.c_decompress/self.frame
            avg_cloud_dnntime = self.cloud_dnntime/self.frame
            avg_fps = self.frame/(time3 - self.start_time)
            t_fps = round(1/(time3-time1),3)
            img_to_draw = self.bboxes_draw_on_img_cv(img_data, bboxes, scores, labels, self.frame, avg_fps)

            self.logger.debug('----------------------------------------------')
            self.logger.debug('Frames,Tstart,Tend,A.PreDNNAndTrans, A.Decompress,A.CloudDnntime, A.FPS')
            self.logger.debug('{},{},{},{},{},{},{}'.format(
                self.frame, time1, time3,avg_prednn_and_trans,avg_decompress,avg_cloud_dnntime, avg_fps))

            self.logger.debug('Frame,PreDNNAndTrans,Decompress,CloudDnntime,FPS')
            self.logger.debug('{},{},{},{},{}'.format(self.frame,t_prednn_and_trans,t_decompress,t_cloud_dnntime,t_fps))

            time4 = time.time()
            # Write timestamps to the time_logger file.
            if self.frame <= 1:
                per_frame_time_diff = round(-1,5)
            else:
                per_frame_time_diff = round(time4 - self.prev_tend,5)

            out.write(img_to_draw)
            # plt.imshow(img_to_draw)
            # plt.show()

            # cv2.imshow('frame', img_to_draw)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     return

            self.time_logger.info('{},{}'.format(self.frame, per_frame_time_diff))
            self.prev_tend = time4
            self.frame +=1

        except:
            self.logger.info('Video ended.')
            return

        return


    def run_one_server(self, input_q, scale, zero_point, img_data_shape):
        conv = input_q
        all_classes, all_scores, all_bboxes = self.cloud_dnn.launch(self.sess_cloud, conv, img_data_shape)
        return all_classes, all_scores, all_bboxes

    def run_one_quantized_server(self,input_q, scale, zero_point, img_data_shape):
        # -----  Cloud Inference code ------------------
        # 2) DeQuantize conv on the cloud side
        quantizer = Quantize()
        input_discrete = self.dequantize(quantizer, input_q, scale, zero_point)
        all_classes, all_scores, all_bboxes = self.cloud_dnn.launch(self.sess_cloud, input_discrete, img_data_shape)
        return all_classes, all_scores, all_bboxes

    def dequantize(self, quantizer, input_q, scale, zero_point):
            return quantizer.linear_dequantize(input_q, scale, zero_point)

    def save_frozen_pb(self):
        cloud_dnn = CloudDNN(False, self.yolo_config)
        with tf.Session(graph=cloud_dnn.graph) as sess2:
            cloud_dnn.load_checkpoint(sess2)
            cloud_dnn.save_frozen_pb(sess2)
            return

    def bboxes_draw_on_img_cv(self, img, bboxes, scores, classes, frame_number, fps):
        text_thickness = 1
        line_type = 1
        colors = [(255, 255, 0), (0, 255, 255), (0, 0, 255), (0, 255, 255), (255, 0, 255), (255, 255, 0),
                  (255, 255, 255), (0, 128, 128), (128, 0, 128), (128, 128, 0), (128, 128, 128), (64, 0, 64)]
        class_names = ['person', 'face']
        thickness = 2

        max_score = 0.0
        max_index = -1
        for ix in range(bboxes.shape[0]):
            bbox = bboxes[ix]

            # Draw bounding boxes
            p1 = (int(bbox[0]), int(bbox[1]))
            p2 = (int(bbox[2]), int(bbox[3]))

            cv2.rectangle(img, p1[::-1], p2[::-1], colors[classes[ix]], thickness)

            # Draw text
            # s = '%s/%.1f' % (class_names[classes[ix]], scores[ix]*100)
            s = '%.1f' % (scores[ix] * 100)
            cv2.putText(img, s, (p1[1], p1[0] + 20), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), text_thickness,
                        line_type)

        txt_type = 'AUTO-SPLIT: #Frame:{}'.format(frame_number)
        # txt_type = 'AUTO-SPLIT: #Frame:{} FPS: {}'.format(frame_number,fps)

        cv2.putText(img, txt_type, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2, 1)

        return img


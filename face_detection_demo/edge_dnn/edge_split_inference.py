from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os
import time
import cv2
import numpy as np
import pathlib
import base64

import pickle
import pprint
import tensorflow as tf #pylint: disable=import-error
import xmlrpc.client

from config_person import YoloConfig
from yolo3_split_model import EdgeDNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from linear_quantization import Quantize
import csv
import subprocess
# Import libraries
import numpy as np  # linear algebra
from PIL import Image
# from bitstruct import pack
from bitstring import BitArray
import bitarray
from compress.compress_act_2 import compress as act_compression
import math
from itertools import groupby
import cv2
import numpy as np
import socket
import sys
import pickle
import struct
import logging

class YoloSplitInference(object):
    def __init__(self, ipaddr='localhost', port='8000'):

        # Set Logger
        logger = logging.getLogger('EdgeDNN')
        logger.setLevel(logging.INFO)
        logfile = pathlib.Path("edge.log")
        if os.path.isfile(logfile):
            logfile.unlink()

        fh = logging.FileHandler('edge.log')
        fh.setLevel(logging.INFO)
        logger.addHandler(fh)
        formatter = logging.Formatter('%(asctime)s-%(name)s-%(levelname)s-%(message)s')
        fh.setFormatter(formatter)
        logger.addHandler(fh)
        self.logger = logger

        self.ipaddr = ipaddr
        self.port = port
        # config = tf.ConfigProto(allow_soft_placement=True) # , log_device_placement=True
        config = tf.ConfigProto(device_count={'CPU': 1}, intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1) #, log_device_placement=True

        # config.gpu_options.allow_growth = True
        self.yolo_config = YoloConfig()
        self.load_checkpoint(self.yolo_config)
        self.time_all = 0.0
        self.frame_count = 1
        self.clientsocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.clientsocket.connect((ipaddr, port))
        self.start_time = time.time()
        self.prev_tend = self.start_time
        self.c_prednn =0
        self.c_compress =0
        self.c_trans =0
        return

    def __del__(self):
        self.sess_edge.close()

    def load_checkpoint(self, yolo_config):
        self.edge_dnn = EdgeDNN(False, yolo_config)
        self.sess_edge = tf.Session(graph=self.edge_dnn.graph)
        self.edge_dnn.load_checkpoint(self.sess_edge)
        return

    def run_many(self, dir_name, is_quantized=False, num_bits=None, is_cloud_only=False, act_compress=False):
        if is_cloud_only:
            run_one_fn = self.run_one_cloud_only
        else:
            if is_quantized:
                run_one_fn = self.run_one_quantized
            else:
                run_one_fn = self.run_one

        if dir_name != '':
            dir_basename = os.path.basename(dir_name)
            image_out = os.path.join('./result', dir_basename)
        else:
            image_out = './result'
        if not os.path.exists(image_out):
            os.mkdir(image_out)

        count = 0
        time_all = 0.0

        image_list = os.listdir(dir_name)
        self.logger.debug('name,shape,time, sparsity')
        for image_file in image_list:
            if image_file.endswith('.jpg') or image_file.endswith('.JPG') or image_file.endswith(
                    '.jpeg') or image_file.endswith('.JPEG') or image_file.endswith('.png') or image_file.endswith(
                    '.PNG') or image_file.endswith('.bmp'):
                count += 1
                # print('image_file : ', image_file)
                img_data = cv2.imread(os.path.join(dir_name, image_file))

                time_start = time.time()
                input_q, scale, zero_point, image_data_shape = run_one_fn(img_data, num_bits)
                scale = float(scale)
                zero_point = float(zero_point)
                input_q_int = input_q.astype(int)
                # input_q_int_list = input_q_int.tolist()
                # Check activation sparsity
                # yy = list(input_q_int.flatten())

                xx = input_q.flatten().astype(int).tolist()
                # Check activation sparsity
                sparsity = round(xx.count(0) / len(xx), 3)
                if act_compress:
                    time1 = time.time()
                    bitarray_x = BitArray().join(BitArray(uint=x, length=num_bits) for x in xx)
                    # # Method 1
                    # x_bytes = bitarray_x.tobytes()
                    # xx_encoded = base64.b64encode(x_bytes)
                    # xx = xx_encoded.decode('ascii')

                    # Method 2
                    time2 = time.time()
                    xx = bitarray_x.tobytes().hex()
                    time3 = time.time()
                    to_bitarray = time2 - time1
                    to_bytes = time3 - time2

                    self.logger.debug('to_bitarray:{}, to_bytes:{}'.format(to_bitarray, to_bytes))

                # yy = np.array(xx)
                # yy.resize(input_q.shape)
                # yy_encoded = encode_list(yy)
                # yy_KB = len(yy)*num_bits/(8*1024)
                # print('out_act: size: {} KB'.format(len(yy)/2/1024))

                # act_file_name = os.path.join('out_act/',image_file.split('.jpg')[0] + '.csv')
                # with open(act_file_name, 'w') as f:
                #     write = csv.writer(f)
                #     write.writerow(yy)
                # f.close()

                # yy = [int(i) for i in yy]
                # sparsity = round(yy.count(0) / len(yy),3)

                with xmlrpc.client.ServerProxy("http://localhost:8000/", allow_none=True) as proxy:
                    proxy.split_inference_server(xx, scale,
                                                 zero_point, image_data_shape, is_quantized,
                                                 image_file, input_q.shape, act_compress, num_bits)

                total_time = round(time.time() - time_start,2)
                time_all += total_time
                # self.logger.debug('{},{},{}, {} '.format(image_file, image_data_shape,total_time, sparsity))

        self.logger.debug(' total images,{},time,{}'.format(count, time_all))

    def send_rpc(self, xx, scale, zero_point, image_data_shape, is_quantized, input_q_shape,
                 act_compress, num_bits):
        ip_addr = 'http://{}:{}'.format(self.ipaddr,self.port)
        with xmlrpc.client.ServerProxy(ip_addr, allow_none=True) as proxy:
            proxy.split_inference_server(xx, scale,
                                         zero_point, image_data_shape, is_quantized,
                                         self.frame_count, input_q_shape, act_compress, num_bits)
        return


    def send_socket(self, xx, scale, zero_point, image_data_shape, is_quantized, input_q_shape,
                     act_compress, num_bits):

        def send_split_act(xx, scale, zero_point, image_data_shape, is_quantized,
                           frame_count, input_q_shape, act_compress, num_bits):
            # --- Prepare values
            HEADERSIZE = 10
            len_ip_shape = len(input_q_shape)
            if len_ip_shape == 4:
                (B, H, W, C) = input_q_shape
            else:
                (H, W, C) = input_q_shape
                B=0

            if num_bits is None:
                num_bits = 0

            (HI, WI, CI) = image_data_shape
            # --- Prepare data

            xx_pickle = pickle.dumps(xx)
            msg_pickle_header = bytes(f'{len(xx_pickle):<{HEADERSIZE}}', 'utf-8')

            other_vars = struct.pack('ddIIII?IIII?I', scale, zero_point, B, H, W, C, is_quantized, frame_count,
                                     HI, WI, CI, act_compress, num_bits)

            msg_other_vars_header = bytes(f'{len(other_vars):<{HEADERSIZE}}', 'utf-8')
            msg = msg_other_vars_header + msg_pickle_header + other_vars + xx_pickle
            return msg

        msg = send_split_act(xx, scale, zero_point, image_data_shape, is_quantized,
                             self.frame_count, input_q_shape, act_compress, num_bits)
        MAXMSG=1
        for i in range(MAXMSG):
            self.clientsocket.sendall(msg)
            data = self.clientsocket.recv(1024)
            recv_msg = data.decode('utf-8')
            recv_frame = int(recv_msg.split(',')[2])
            assert recv_frame == self.frame_count,'Sent and received frames do not match, sent:{}, ' \
                                                    'received:{}'.format(self.frame_count,recv_frame)
            self.logger.debug('Received: {}'.format(data.decode('utf-8')))
        return


    def run_many_video(self, cap, is_quantized=False, num_bits=None, is_cloud_only=False, act_compress=False):
        act_compress_2 = False
        is_rpc=False
        time1 = time.time()
        if self.frame_count <=1:
            self.start_time = time1
            self.prev_tend = time1

        if is_cloud_only:
            run_one_fn = self.run_one_cloud_only
        else:
            if is_quantized:
                run_one_fn = self.run_one_quantized
            else:
                run_one_fn = self.run_one

        try:

            # self.logger.debug('----------------------------------------------')
            # self.logger.debug('Frame #: ', self.frame_count)

            cap.isOpened()
            ret, frame = cap.read()
            img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # cv2.imshow('frame', img)
            # if cv2.waitKey(1) & 0xFF == ord('q'):
            #     return

            # img_data = cv2.imread(os.path.join(dir_name, image_file))


            input_q, scale, zero_point, image_data_shape = run_one_fn(img_data, num_bits)
            scale = float(scale)
            zero_point = float(zero_point)
            input_q_shape = input_q.shape
            time2 = time.time()
            if act_compress:
                (B, H, W, C) = input_q.shape
                input_q.resize(H, W, C)
                C_by_2 = int(math.ceil(C / 2))
                input_q = input_q.astype(np.uint8)
                input_packed_q = act_compression(input_q, H, W, C_by_2, num_bits)
                xx = input_packed_q.flatten().astype(int).tolist()

            elif act_compress_2:
                # Works with only RPC..
                if not is_rpc:
                    raise ValueError('act_compress_2 works with only rpc')
                xx = input_q.flatten().astype(int).tolist()
                bitarray_x = BitArray().join(BitArray(uint=x, length=num_bits) for x in xx)
                xx = bitarray_x.tobytes().hex()
                input_packed_q = None
            else:
                # fake compression starts .. to even out 8-bit vs 4-bit data transmission..
                # 4-bit data requires packing, whereas 8-bit can be transmitted directly
                H=36
                W=64
                C=256
                C_by_2 = 128
                vol  = H*W*C
                # Initialization takes extra time..
                random_array = np.random.randint(0, 15, size=vol, dtype=np.uint8)
                random_array.resize(H, W, C)
                _ = act_compression(random_array, H, W, C_by_2, 4)
                # -- fake compression ends --
                xx = input_q.flatten().astype(int).tolist()
                input_packed_q = input_q

            # Check activation sparsity
            # sparsity = round(xx.count(0) / len(xx), 3)
            time3 = time.time()
            if is_rpc:
                self.send_rpc(xx, scale, zero_point, image_data_shape, is_quantized, input_q_shape,
                     act_compress, num_bits)
            else:
                self.send_socket(input_packed_q, scale, zero_point, image_data_shape, is_quantized, input_q_shape,
                     act_compress, num_bits)


            time4 = time.time()

            avg_delay = self.prev_tend - time1
            self.prev_tend = time4
            self.c_prednn +=  time2 - time1
            self.c_compress += time3 - time2
            self.c_trans += time4 - time3

            avg_prednn = self.c_prednn/self.frame_count
            avg_compress = self.c_compress/self.frame_count
            avg_trans = self.c_trans/self.frame_count
            avg_fps = self.frame_count/(self.prev_tend - self.start_time)

            self.logger.debug('----------------------------------------------')
            self.logger.debug('Frames,Tstart,Tend,A.delay,A.PreDNN,A.Compress,A.Trans,A.FPS')
            self.logger.debug('{},{},{},{},{},{},{},{}'.format(self.frame_count,time1,time4,avg_delay,avg_prednn,avg_compress,
                                              avg_trans,avg_fps))


            total_time = time4 - time1
            self.time_all += total_time
            self.frame_count += 1
        except:
            self.logger.warning('Video ended.')
            return

        # print('# Frames: {}, Cumm. time: {}'.format(self.frame_count, self.time_all))
        return

    def run_one(self, img_data, num_bits=None):
        img_data_shape = img_data.shape
        conv, _ = self.edge_dnn.launch(self.sess_edge, img_data)
        return conv, 0, 0, img_data_shape

    def run_one_cloud_only_processed(self, img_data, num_bits=None):
        def preprocess(image, input_shape):
            '''
            resize image with unchanged aspect ratio using padding by opencv
            '''
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            h, w, _ = image.shape

            input_h, input_w = input_shape
            scale = min(float(input_w) / float(w), float(input_h) / float(h))
            nw = int(w * scale)
            nh = int(h * scale)

            image = cv2.resize(image, (nw, nh))

            new_image = np.zeros((input_h, input_w, 3), np.float32)
            new_image.fill(128)
            bh, bw, _ = new_image.shape
            # new_image[:nh, :nw, :] = image
            new_image[int((bh - nh) / 2):(nh + int((bh - nh) / 2)), int((bw - nw) / 2):(nw + int((bw - nw) / 2)),
            :] = image

            new_image /= 255.
            new_image = np.expand_dims(new_image, 0)  # Add batch dimension.
            return new_image

        img_data = preprocess(img_data, self.yolo_config.input_shape)
        assert img_data.min() >=0,'Current assumption activations after relu. If not true modify ' \
                                  'code to use signed bits instead'
        return img_data, 0, 0, self.yolo_config.input_shape

    def run_one_cloud_only(self, img_data, num_bits=None):
        assert img_data.min() >= 0, 'Current assumption activations after relu. If not true modify ' \
                                    'code to use signed bits instead'
        return img_data, 0, 0, img_data.shape

    def run_one_quantized(self, img_data, num_bits):
        # -----  Edge Inference code ------------------
        img_data_shape = img_data.shape
        conv, conv_index = self.edge_dnn.launch(self.sess_edge, img_data)
        sat_min = conv.min()
        assert sat_min == 0, 'Current assumption activations after relu. If not true modify code to use signed bits ' \
                             'instead'
        sat_max = conv.max()
        # 1) Quantize conv -- on the edge side
        quantizer = Quantize()
        input_q, scale, zero_point = self.quantize(quantizer, conv, num_bits=num_bits, sat_min=sat_min, sat_max=sat_max)
        return input_q, scale, zero_point, img_data_shape

    def bboxes_draw_on_img_cv(self, img, bboxes, scores, classes):
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

        return img

    def quantize(self, quantizer, input, num_bits, sat_min=-1, sat_max=1, scale=None, zero_point=None, clipping=None):
        if scale is None or zero_point or None or clipping is None:
            scale, zero_point = quantizer.linear_quantize_params(num_bits, sat_min, sat_max)
            input_q = quantizer.linear_quantize(input, scale, zero_point)
        else:
            # TODO: Add clipping logic. Clip vectors beyong clipping point.
            #  for act: (0,clipping), for weights = currently not supported.
            # input = input > clipping
            scale_arr = np.full_like(num_bits, scale[0], dtype='float32')
            input_q = quantizer.linear_quantize(input, scale_arr, zero_point)

        return input_q, scale, zero_point

    def save_frozen_pb(self):
        edge_dnn = EdgeDNN(False, self.yolo_config)
        with tf.Session(graph=edge_dnn.graph) as sess1:
            edge_dnn.load_checkpoint(sess1)
            edge_dnn.save_frozen_pb(sess1)
            return

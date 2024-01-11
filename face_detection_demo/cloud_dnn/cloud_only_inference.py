
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

import tensorflow as tf

from config_person import YoloConfig
from config_person import YoloConfig
from yolo3_split_model import CloudDNN, EdgeDNN
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from linear_quantization import Quantize
from bitstruct import unpack
import pathlib

class Yolo:
    def __init__(self, logger, test_dir_name, result_dir):
        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        config.gpu_options.allow_growth = True
        self.yolo_config = YoloConfig()
        self.frame =1
        # Run one test case.
        self.load_checkpoint()
        self.start_time = 0.0
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

        time_logger_filename = os.path.join(self.result_dir,"cloud_only_timestamp_{}.csv".format(name_prefix))
        logfile = pathlib.Path(time_logger_filename)
        if os.path.isfile(logfile):
            logfile.unlink()

        time_fh = logging.FileHandler(time_logger_filename)
        time_fh.setLevel(logging.INFO)
        time_logger.addHandler(time_fh)
        time_logger.addHandler(time_fh)
        self.time_logger = time_logger
        return

    def launch(self, img_data, img_data_shape):
        # self.logger.debug('image_data_shape: {}'.format(img_data.shape))
        conv, conv_index = self.edge_dnn.launch(self.sess_edge, img_data)
        # self.logger.debug('CONV IDX: {}'.format(conv_index))
        all_classes, all_scores, all_bboxes = self.cloud_dnn.launch(self.sess_cloud, conv, img_data_shape)
        return all_classes, all_scores, all_bboxes

    def load_checkpoint(self):
        self.edge_dnn = EdgeDNN(False, self.yolo_config)
        self.sess_edge = tf.Session(graph=self.edge_dnn.graph)
        self.edge_dnn.load_checkpoint(self.sess_edge)

        self.cloud_dnn = CloudDNN(False, self.yolo_config)
        self.sess_cloud = tf.Session(graph=self.cloud_dnn.graph)
        self.cloud_dnn.load_checkpoint(self.sess_cloud)
        return

    def run_many_server(self, input_q_int_list, scale, zero_point, image_data_shape, is_quantized, image_file_name,
                        input_shape, act_compress, act_bitwidth):

        recovered_input_q_int = np.array(input_q_int_list)
        recovered_input_q_int.resize(image_data_shape)

        # recovered_input_q = recovered_input_q_int.astype(int)
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
        time_start = time.time()

        ip_data_type = img_data.dtype
        recovered_input_q = recovered_input_q_int.astype(ip_data_type)
        labels, scores, bboxes = self.launch(recovered_input_q, image_data_shape)
        total_time = time.time() - time_start
        self.logger.debug('{},{},{} '.format(image_file, img_data.shape, total_time))
        img_to_draw = self.bboxes_draw_on_img_cv(img_data, bboxes, scores, labels)
        plt.imshow(img_to_draw)
        plt.show()

        # cv2.imwrite(os.path.join(image_out, image_file), img_to_draw)
        final_image_path = os.path.join(image_out, image_file_name)
        plt.imsave(final_image_path, img_to_draw)
        return

    def run_incloud_many_video_server(self, cap):
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        # total_frames = 10
        for i in range(1,total_frames):
            try:
                time_start = time.time()
                cap.isOpened()
                ret, frame = cap.read()
                # img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img_data = frame
                ip_data_type = img_data.dtype

                # labels, scores, bboxes = self.launch(recovered_input_q, image_data_shape)
                labels, scores, bboxes = self.launch(img_data, img_data.shape)
                img_to_draw = self.bboxes_draw_on_img_cv(img_data, bboxes, scores, labels,0)

                # plt.imshow(img_to_draw)
                # plt.show()

                # cv2.imshow('frame', img_to_draw)
                # if cv2.waitKey(1) & 0xFF == ord('q'):
                #     return
                time_end  = time.time()
                # self.logger.debug('----------------------------------------------')
                # self.logger.debug('Frame #: {}  Cumm. DNN run time: {} '.format(i, time_end - self.start_time))
                frame+=1
            except:
                self.logger.debug('Video ended.')
                return

        return


    def run_many_video_server(self, cap, out, xx_encoded, scale, zero_point,
                              image_data_shape, is_quantized,
                              image_file_name, input_shape, act_compress, act_bitwidth):

        try:
            time1 = time.time()
            if self.frame <= 1:
                self.start_time = time1
                self.prev_tend = time1

            cap.isOpened()
            ret, frame = cap.read()
            img_data = frame
            # img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            ip_data_type = img_data.dtype



            def fake_decompress():
                H=36
                W=64
                C=256
                C_by_2 = 128
                num_bits=4
                vol = H*W*C_by_2
                input_packed = np.random.randint(0, 15, size=vol, dtype=np.uint8)
                input_packed.resize(H,W,C_by_2)

                recovered_input = np.zeros([H, W, C], dtype=np.uint8)
                for i in range(0, C_by_2):
                    l_x = np.left_shift(input_packed[:, :, i], num_bits)
                    y = np.right_shift(l_x, num_bits)
                    x = np.right_shift(input_packed[:, :, i], num_bits)
                    recovered_input[:, :, 2 * i] = x
                    recovered_input[:, :, 2 * i + 1] = y

                return recovered_input

            # Since, in proof of concept. Int-4 data type does not exist in Python. It leads to an extra overhead
            # of packing 4-bits into 8-bit data. This overhead will not exist if the edge device supports INT-4 data
            # type. Hence, for cloud-only solution same overhead is added for a fair comparison.

            _ = fake_decompress()
            recovered_input_q = np.array(xx_encoded)
            recovered_input_q = recovered_input_q.astype(ip_data_type)
            recovered_input_q.resize(input_shape)
            time2 = time.time()
            labels, scores, bboxes = self.launch(recovered_input_q, image_data_shape)
            time3 = time.time()

            t_prednn_and_trans = time1  - self.prev_tend
            self.c_transmission += t_prednn_and_trans

            t_decompress = time2 - time1
            self.c_decompress += t_decompress

            t_cloud_dnntime = time3 - time2
            self.cloud_dnntime += t_cloud_dnntime

            avg_prednn_and_trans = self.c_transmission/self.frame
            avg_decompress = self.c_decompress/self.frame
            avg_cloud_dnntime = self.cloud_dnntime/self.frame
            avg_fps = self.frame/(time3 - self.start_time)
            t_fps = round(1 / (time3 - time1), 3)
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
            self.frame += 1
        except:
            self.logger.info('Video ended.')
            return

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

        txt_type = 'CLOUD-ONLY: #Frame:{}'.format(frame_number)
        # txt_type = 'CLOUD-ONLY: #Frame:{} FPS: {}'.format(frame_number, fps)
        cv2.putText(img, txt_type, (30,40), cv2.FONT_HERSHEY_SIMPLEX, 1, (148, 0, 211), 2, 1)
        return img

    def __del__(self):
        self.sess_cloud.close()
        self.sess_edge.close()



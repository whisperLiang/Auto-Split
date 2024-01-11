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
import scipy.sparse
from test_sparsity.huffman import HuffmanCoding
import sys
import csv
import subprocess
# Import libraries
import numpy as np  # linear algebra
from PIL import Image

from itertools import groupby


def encode_list(s_list):
    # [[len(list(group)), key] for key, group in groupby(s_list)]
    run_length = []
    for key, group in groupby(s_list):
        run_length.append(len(list(group)))
        run_length.append(key)
    return run_length

class YoloSplitInference(object):
    def __init__(self):

        # config = tf.ConfigProto(allow_soft_placement=True) # , log_device_placement=True
        config = tf.ConfigProto(device_count={'CPU': 1}, intra_op_parallelism_threads=1,
                                inter_op_parallelism_threads=1) #, log_device_placement=True

        # config.gpu_options.allow_growth = True
        self.yolo_config = YoloConfig()
        self.load_checkpoint(self.yolo_config)
        return

    def __del__(self):
        self.sess_edge.close()

    def load_checkpoint(self, yolo_config):
        self.edge_dnn = EdgeDNN(False, yolo_config)
        self.sess_edge = tf.Session(graph=self.edge_dnn.graph)
        self.edge_dnn.load_checkpoint(self.sess_edge)
        return

    def run_many(self, dir_name, is_quantized=False, num_bits=None, is_cloud_only=False):
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
        print('name,shape,time, sparsity')
        for image_file in image_list:
            if image_file.endswith('.jpg') or image_file.endswith('.JPG') or image_file.endswith(
                    '.jpeg') or image_file.endswith('.JPEG') or image_file.endswith('.png') or image_file.endswith(
                    '.PNG') or image_file.endswith('.bmp'):
                count += 1
                # print('image_file : ', image_file)
                img_data = cv2.imread(os.path.join(dir_name, image_file))

                time_start = time.time()
                input_q, scale, zero_point, image_data_shape = run_one_fn(img_data, num_bits)
                (b,c,w,h) = input_q.shape

                for i in range(0,c,3):
                    i_q = input_q[0,i:i+3,:,:]
                    # rescale -- change from [-8,7] to [0,15]
                    iq_new = i_q + 8
                    iq_new = iq_new.astype(np.uint8)
                    iq_new.resize(64, 256, 3)
                    img = Image.fromarray(iq_new, mode="RGB")
                    file_name = image_file.split('.jpg')[0]
                    img_dir = os.path.join('out_act/', file_name)
                    pathlib.Path(img_dir).mkdir(parents=True, exist_ok=True)
                    img_path_out = os.path.join(img_dir, '{}.jpg'.format(i))
                    # save as jpeg.
                    img.save(img_path_out, "JPEG", quality=100)

                # Bash read filenames and add file sizes
                out_act_dir = 'out_act/'
                img_dir_names = os.listdir(out_act_dir)
                for img_dirs in img_dir_names:
                    new_dir = os.path.join(out_act_dir,img_dirs)
                    # os.chdir(new_dir)
                    # find . -type f -printf "%s %p\n"
                    file_list = os.listdir(new_dir)
                    total_size = 0
                    for file in file_list:
                        if not '.jpg' in file:
                            continue

                        file_size = os.path.getsize( os.path.join(new_dir,file))
                        total_size += file_size

                    print('{},{},KB'.format(img_dirs,round(total_size/1024,2)))
                    # os.chdir(out_act_dir)




                scale = float(scale)
                zero_point = float(zero_point)
                input_q_int = input_q.astype(int)
                input_q_int_list = input_q_int.tolist()
                # Check activation sparsity
                yy = list(input_q_int.flatten())
                yy_encoded = encode_list(yy)
                yy_KB = len(yy)*num_bits/(8*1024)
                print('out_act: size: {} KB'.format(len(yy)/2/1024))
                # write to csv file..
                # np.resize(input_q, (36, 64, 256))
                # input_q[0, 0:3, :, :].shape

                pathlib.Path('out_act_raw/').mkdir(parents=True, exist_ok=True)
                act_file_name = os.path.join('out_act_raw/',image_file.split('.jpg')[0] + '.csv')
                with open(act_file_name, 'w') as f:
                    write = csv.writer(f)
                    write.writerow(yy)
                f.close()

                bashCommand = "tar -zcvf {}.tar.gz {}".format(act_file_name,act_file_name)
                process = subprocess.Popen(bashCommand.split(), stdout=subprocess.PIPE)
                output, error = process.communicate()


                h = HuffmanCoding(act_file_name)
                output_path = h.compress()
                print("Compressed file path: " + output_path)

                decom_path = h.decompress(output_path)
                print("Decompressed file path: " + decom_path)

                # num_dict = {}
                #
                # for num,val in yy_encoded:
                #     if num in num_dict:
                #         num_dict[num] +=1
                #     else:
                #         num_dict[num] = 1
                #
                # val_dict = {}
                #
                # for num, val in yy_encoded:
                #     if val in val_dict:
                #         val_dict[val] += 1
                #     else:
                #         val_dict[val] = 1

                yy = [int(i) for i in yy]
                sparsity = round(yy.count(0) / len(yy),3)
                out_file_path = (os.path.join('out_act/', image_file))


                # input_q.save(out_file_path, "JPEG", quality=100)


                # with xmlrpc.client.ServerProxy("http://localhost:8000/") as proxy:
                #     proxy.split_inference_server(input_q_int_list, scale,
                #                                  zero_point, image_data_shape, is_quantized,
                #                                  image_file, input_q.shape, 'int')

                total_time = round(time.time() - time_start,2)
                time_all += total_time
                print('{},{},{}, {} '.format(image_file, image_data_shape,total_time, sparsity))

        print(' total images,{},time,{}'.format(count, time_all))

    def run_many_video(self, cap, is_quantized=False, num_bits=None, is_cloud_only=False):

        if is_cloud_only:
            run_one_fn = self.run_one_cloud_only
        else:
            if is_quantized:
                run_one_fn = self.run_one_quantized
            else:
                run_one_fn = self.run_one

        count = 0
        time_all = 0.0

        try:
            cap.isOpened()
            ret, frame = cap.read()
            img_data = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            # cv2.imshow('frame', img)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                return

            count += 1
            print('----------------------------------------------')
            print('Frame #: ', count)

            # img_data = cv2.imread(os.path.join(dir_name, image_file))

            time_start = time.time()
            input_q, scale, zero_point, image_data_shape = run_one_fn(img_data, num_bits)
            scale = float(scale)
            zero_point = float(zero_point)
            input_q_int = input_q.astype(int)
            input_q_int_list = input_q_int.tolist()
            # Check activation sparsity
            yy = list(input_q_int.flatten())
            yy = [int(i) for i in yy]
            sparsity = round(yy.count(0) / len(yy), 3)

            with xmlrpc.client.ServerProxy("http://localhost:8000/") as proxy:
                proxy.split_inference_server(input_q_int_list, scale,
                                             zero_point, image_data_shape, is_quantized,
                                             count, input_q.shape, 'int')

            total_time = round(time.time() - time_start, 2)
            time_all += total_time
            print('{},{},{}, {} '.format(count, image_data_shape, total_time, sparsity))
        except:
            print('Video ended.')
            return

        return

    def run_one(self, img_data, num_bits=None):
        img_data_shape = img_data.shape
        conv, _ = self.edge_dnn.launch(self.sess_edge, img_data)
        return conv, 0, 0, img_data_shape

    def run_one_cloud_only(self, img_data, num_bits=None):
        img_data_shape = img_data.shape
        return img_data, 0, 0, img_data_shape

    def run_one_quantized(self, img_data, num_bits):
        # -----  Edge Inference code ------------------
        img_data_shape = img_data.shape
        conv, conv_index = self.edge_dnn.launch(self.sess_edge, img_data)
        sat_min = conv.min()
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

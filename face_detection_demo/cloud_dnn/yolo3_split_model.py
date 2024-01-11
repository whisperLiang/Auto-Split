# -*- coding: utf-8 -*-


"""
Yolo3 model
"""


import os
import logging
import numpy as np
import tensorflow as tf
import cv2
from tensorflow.python.framework import graph_util



class EdgeDNN:
    def __init__(self, is_training, config):
        """
        Introduction
        ------------
            初始化函数
        ----------
        """

        logging.info('EdgeDNN: is_training: %s' % is_training)
        logging.info('EdgeDNN: model dir: %s' % config.model_dir)
        logging.info('EdgeDNN: input_shape: (%d, %d)' % (config.input_shape[0], config.input_shape[1]))
        logging.info('EdgeDNN: learning rate: %f' % config.learning_rate)

        self.is_training = is_training
        self.model_dir = config.model_dir
        self.norm_epsilon = config.norm_epsilon
        self.norm_decay = config.norm_decay
        self.obj_threshold = config.obj_threshold
        self.nms_threshold = config.nms_threshold
        self.net_type = config.net_type
        self.model_dir = config.model_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.anchors = np.array([float(x) for x in config.anchors]).reshape(-1, 2)
        self.class_names = config.classes
        self.num_classes = len(self.class_names)
        self.input_shape = config.input_shape

        print("EdgeDNN: anchors : ", self.anchors)
        print("EdgeDNN: class_names : ", self.class_names)



        self.graph = tf.Graph()
        with self.graph.as_default():
            if is_training:
                self.images = tf.placeholder(shape=[None, None, None, 3], dtype=tf.float32, name='images')
            else:
                # self.images = tf.placeholder(shape=[1, 224, 416, 3], dtype=tf.float32, name='images')
                self.images = tf.placeholder(shape=[None, self.input_shape[0], self.input_shape[1], 3],
                                             dtype=tf.float32,
                                             name='images')

            self.image_shape = tf.placeholder(dtype=tf.int32, shape=(2,), name='shapes')
            self._darknet53_edge(self.images, self.is_training, self.norm_decay, self.norm_epsilon)

    def load_checkpoint(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                pretrain_saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(self.model_dir)
                pretrain_saver.restore(sess, ckpt.model_checkpoint_path)
                return

    def save_frozen_pb(self, sess):
        input_graph_def = sess.graph.as_graph_def()
        # nodes = [n.name + ' => ' + n.op for n in input_graph_def.node if n.op in ('Placeholder')]
        # print(nodes)
        frozen_pb_dir = os.path.join(self.model_dir,'frozen_pb')
        pbtxt_filename = 'yolo3_edge_{}.pbtxt'.format(self.net_type)
        pb_filename = 'yolo3_edge_{}.pb'.format(self.net_type)
        pb_filepath = os.path.join(frozen_pb_dir, pb_filename)

        # 1. Writes the graph without the parameters.
        tf.train.write_graph(input_graph_def, frozen_pb_dir, pbtxt_filename, True)
        # input_nodes = []
        # for node in graph_def.node:
        #     if node.op in ('Placeholder'):
        #         tmp = node.name + " => " + node.op
        #         input_nodes.append(tmp)
        # print(input_nodes)

        # Method 2: a) Serialize, b) write
        output_tensors = ['darknet53/add_10']
        print ('output_tensors : ', output_tensors)
        # output_tensors = [t.op.name for t in output_tensors]
        graph = graph_util.convert_variables_to_constants(sess, input_graph_def, output_tensors)
        with tf.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(graph.SerializeToString())

        logging.info("save model as .pb end .......")

    def launch(self, sess, img_data):
        def create_output_fetch(sess):
            '''
            create output fetch tensors
            '''
            output_fetch = sess.graph.get_tensor_by_name('darknet53/add_10:0')
            # output_fetch = sess.graph.get_tensor_by_name('darknet53/conv2d_26/Conv2D')
            return output_fetch

        def create_input_feed(sess, new_image, img_data):
            '''
            create input feed data
            '''

            input_feed = {}

            input_img_data = sess.graph.get_tensor_by_name('images:0')  # input_1 #images
            input_feed[input_img_data] = new_image
            #
            # input_img_shape = sess.graph.get_tensor_by_name('shapes:0')  # Placeholder #shapes
            # input_feed[input_img_shape] = [img_data.shape[0], img_data.shape[1]]

            return input_feed

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

        new_image = preprocess(img_data, self.input_shape)

        with sess.as_default():
            with sess.graph.as_default():
                input_feed = create_input_feed(sess, new_image, img_data)
                output_fetch = create_output_fetch(sess)
                conv = sess.run(output_fetch, feed_dict=input_feed)
                conv_index=27
                return conv, conv_index

    def _batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.997, norm_epsilon=1e-5):
        '''
        Introduction
        ------------
            对卷积层提取的feature map使用batch normalization
        Parameters
        ----------
            input_layer: 输入的四维tensor
            name: batchnorm层的名字
            trainging: 是否为训练过程
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            bn_layer: batch normalization处理之后的feature map
        '''
        bn_layer = tf.layers.batch_normalization(inputs=input_layer,
                                                 momentum=norm_decay, epsilon=norm_epsilon, center=True,
                                                 scale=True, training=training, name=name, fused=True)
        return tf.nn.relu(bn_layer)  # leaky_relu not supported by D-chip right now
        # return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        """
        Introduction
        ------------
            使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
            经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
            根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
            比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
            因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            strides: 卷积步长
            name: 卷积层名字
            trainging: 是否为训练过程
            use_bias: 是否使用偏置项
            kernel_size: 卷积核大小
        Returns
        -------
            conv: 卷积之后的feature map
        """
        if strides > 1:  # modified 0327
            # 在输入feature map的长宽维度进行padding
            inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        conv = tf.layers.conv2d(inputs=inputs, filters=filters_num,
                                kernel_size=kernel_size, strides=[strides, strides],
                                padding=('SAME' if strides == 1 else 'VALID'),  # padding = 'SAME', #
                                use_bias=use_bias,
                                name=name)  # , kernel_initializer = tf.contrib.layers.xavier_initializer()
        return conv

    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True, norm_decay=0.997,
                        norm_epsilon=1e-5):
        """
        Introduction
        ------------
            Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核，使用1x1是为了减少channel的维度
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            trainging: 是否为训练过程
            blocks_num: block的数量
            conv_index: 为了方便加载预训练权重，统一命名序号
            weights_dict: 加载预训练模型的权重
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            inputs: 经过残差网络处理后的结果
        """

        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name="conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index), training=training,
                                                norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size=1, strides=1,
                                       name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def _darknet53_edge(self, inputs=None, training=False, norm_decay=0.997, norm_epsilon=1e-5):
        if inputs is None:
            inputs = self.images

        """
        Introduction
        ------------
            构建yolo3使用的darknet53网络结构
        Parameters
        ----------
            inputs:       模型输入变量
            weights_dict: 预训练权重
            training:     是否为训练
            norm_decay:   在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            conv:       经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
            route1:     返回第26层卷积计算结果52x52x256, 供后续使用
            route2:     返回第43层卷积计算结果26x26x512, 供后续使用
            conv_index: 卷积层计数，方便在加载预训练模型时使用
        """
        conv_index = 1
        with tf.variable_scope('darknet53') as scope:
            conv = self._conv2d_layer(inputs, filters_num=32, kernel_size=3, strides=1,
                                      name="conv2d_" + str(conv_index))
            conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index),
                                                   training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=64, blocks_num=1,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=128, blocks_num=2,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=256, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        return conv, conv_index

class CloudDNN:

    def __init__(self, is_training, config):
        """
        Introduction
        ------------
            初始化函数
        ----------
        """
        self.config = config
        logging.info('is_training: %s' % is_training)
        logging.info('model dir: %s' % config.model_dir)
        logging.info('input_shape: (%d, %d)' % (config.input_shape[0], config.input_shape[1]))
        logging.info('learning rate: %f' % config.learning_rate)

        self.is_training = is_training
        self.model_dir = config.model_dir
        self.norm_epsilon = config.norm_epsilon
        self.norm_decay = config.norm_decay
        self.obj_threshold = config.obj_threshold
        self.nms_threshold = config.nms_threshold
        self.net_type = config.net_type
        self.model_dir = config.model_dir
        if not os.path.exists(self.model_dir):
            os.mkdir(self.model_dir)

        self.anchors = np.array([float(x) for x in config.anchors]).reshape(-1, 2)
        self.class_names = config.classes
        self.num_classes = len(self.class_names)
        self.input_shape = config.input_shape

        print("anchors : ", self.anchors)
        print("class_names : ", self.class_names)

        self.graph = tf.Graph()
        with self.graph.as_default():
            self._darknet53_cloud(self.is_training, self.norm_decay, self.norm_epsilon)
            return

    def load_checkpoint(self, sess):
        with sess.as_default():
            with sess.graph.as_default():
                pretrain_saver = tf.train.Saver()
                ckpt = tf.train.get_checkpoint_state(self.model_dir)
                pretrain_saver.restore(sess, ckpt.model_checkpoint_path)
                return

    def launch(self, sess, conv, img_data_shape):
        def create_output_fetch(sess):
            '''
            create output fetch tensors
            '''

            output_classes = sess.graph.get_tensor_by_name('output/classes:0')  # classes #concat_19
            output_scores = sess.graph.get_tensor_by_name('output/scores:0')  # scores #concat_18
            output_boxes = sess.graph.get_tensor_by_name('output/boxes:0')  # boxes #concat_17
            output_fetch = [output_classes, output_scores, output_boxes]

            return output_fetch

        def create_input_feed(sess, conv, img_data_shape):
            '''
            create input feed data
            '''
            names = [n.name for n in tf.get_default_graph().as_graph_def().node]
            # nodes = [n.name + ' => ' + n.op for n in tf.get_default_graph().as_graph_def().node if n.op in ('Placeholder')]
            # for op in tf.get_default_graph().get_operations():
            #     print(op.name, op.outputs)

            input_feed = {}

            input_conv_data = sess.graph.get_tensor_by_name('darknet53/input:0')  # input_1 #images
            input_feed[input_conv_data] = conv

            input_img_shape = sess.graph.get_tensor_by_name('shapes:0')  # Placeholder #shapes
            input_feed[input_img_shape] = [img_data_shape[0], img_data_shape[1]]

            # input_conv_index = sess.graph.get_tensor_by_name('shapes:0')  # Placeholder #shapes
            # input_feed[input_img_shape] = [img_data.shape[0], img_data.shape[1]]

            return input_feed

        with sess.as_default():
            with sess.graph.as_default():
                input_feed = create_input_feed(sess, conv, img_data_shape)
                output_fetch = create_output_fetch(sess)
                all_classes, all_scores, all_bboxes = sess.run(output_fetch, feed_dict=input_feed)
                return all_classes, all_scores, all_bboxes

    def save_frozen_pb(self, sess):
        input_graph_def = sess.graph.as_graph_def()
        nodes = [n.name + ' => ' + n.op for n in input_graph_def.node if n.op in ('Placeholder')]
        print(nodes)
        frozen_pb_dir = os.path.join(self.model_dir, 'frozen_pb')
        pbtxt_filename = 'yolo3_cloud_{}.pbtxt'.format(self.net_type)
        pb_filename = 'yolo3_cloud_{}.pb'.format(self.net_type)
        pb_filepath = os.path.join(frozen_pb_dir, pb_filename)

        # 1. Writes the graph without the parameters.
        tf.train.write_graph(input_graph_def, frozen_pb_dir, pbtxt_filename, True)
        # input_nodes = []
        # for node in graph_def.node:
        #     if node.op in ('Placeholder'):
        #         tmp = node.name + " => " + node.op
        #         input_nodes.append(tmp)
        # print(input_nodes)

        # Method 2: a) Serialize, b) write
        # output_classes = sess.graph.get_tensor_by_name('output/classes:0')  # classes #concat_19
        # output_scores = sess.graph.get_tensor_by_name('output/scores:0')  # scores #concat_18
        # output_boxes = sess.graph.get_tensor_by_name('output/boxes:0')  # boxes #concat_17
        output_tensors = ['output/classes','output/scores', 'output/boxes' ]
        print('output_tensors : ', output_tensors)
        # output_tensors = [t.op.name for t in output_tensors]
        graph = graph_util.convert_variables_to_constants(sess, input_graph_def, output_tensors)
        with tf.gfile.GFile(pb_filepath, 'wb') as f:
            f.write(graph.SerializeToString())

        logging.info("save model as .pb end .......")
        return

    def _batch_normalization_layer(self, input_layer, name=None, training=True, norm_decay=0.997, norm_epsilon=1e-5):
        '''
        Introduction
        ------------
            对卷积层提取的feature map使用batch normalization
        Parameters
        ----------
            input_layer: 输入的四维tensor
            name: batchnorm层的名字
            trainging: 是否为训练过程
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            bn_layer: batch normalization处理之后的feature map
        '''
        bn_layer = tf.layers.batch_normalization(inputs=input_layer,
                                                 momentum=norm_decay, epsilon=norm_epsilon, center=True,
                                                 scale=True, training=training, name=name, fused=True)
        return tf.nn.relu(bn_layer)  # leaky_relu not supported by D-chip right now
        # return tf.nn.leaky_relu(bn_layer, alpha = 0.1)

    def _conv2d_layer(self, inputs, filters_num, kernel_size, name, use_bias=False, strides=1):
        """
        Introduction
        ------------
            使用tf.layers.conv2d减少权重和偏置矩阵初始化过程，以及卷积后加上偏置项的操作
            经过卷积之后需要进行batch norm，最后使用leaky ReLU激活函数
            根据卷积时的步长，如果卷积的步长为2，则对图像进行降采样
            比如，输入图片的大小为416*416，卷积核大小为3，若stride为2时，（416 - 3 + 2）/ 2 + 1， 计算结果为208，相当于做了池化层处理
            因此需要对stride大于1的时候，先进行一个padding操作, 采用四周都padding一维代替'same'方式
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            strides: 卷积步长
            name: 卷积层名字
            trainging: 是否为训练过程
            use_bias: 是否使用偏置项
            kernel_size: 卷积核大小
        Returns
        -------
            conv: 卷积之后的feature map
        """
        if strides > 1:  # modified 0327
            # 在输入feature map的长宽维度进行padding
            inputs = tf.pad(inputs, paddings=[[0, 0], [1, 0], [1, 0], [0, 0]], mode='CONSTANT')
        conv = tf.layers.conv2d(inputs=inputs, filters=filters_num,
                                kernel_size=kernel_size, strides=[strides, strides],
                                padding=('SAME' if strides == 1 else 'VALID'),  # padding = 'SAME', #
                                use_bias=use_bias,
                                name=name)  # , kernel_initializer = tf.contrib.layers.xavier_initializer()
        return conv

    def _Residual_block(self, inputs, filters_num, blocks_num, conv_index, training=True, norm_decay=0.997,
                        norm_epsilon=1e-5):
        """
        Introduction
        ------------
            Darknet的残差block，类似resnet的两层卷积结构，分别采用1x1和3x3的卷积核，使用1x1是为了减少channel的维度
        Parameters
        ----------
            inputs: 输入变量
            filters_num: 卷积核数量
            trainging: 是否为训练过程
            blocks_num: block的数量
            conv_index: 为了方便加载预训练权重，统一命名序号
            weights_dict: 加载预训练模型的权重
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            inputs: 经过残差网络处理后的结果
        """

        layer = self._conv2d_layer(inputs, filters_num, kernel_size=3, strides=2, name="conv2d_" + str(conv_index))
        layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index), training=training,
                                                norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        for _ in range(blocks_num):
            shortcut = layer
            layer = self._conv2d_layer(layer, filters_num // 2, kernel_size=1, strides=1,
                                       name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer = self._conv2d_layer(layer, filters_num, kernel_size=3, strides=1, name="conv2d_" + str(conv_index))
            layer = self._batch_normalization_layer(layer, name="batch_normalization_" + str(conv_index),
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            conv_index += 1
            layer += shortcut
        return layer, conv_index

    def yolo_inference(self, features_out, filters_yolo_block, conv_index, num_anchors, num_classes, training=True):
        """
        Introduction
        ------------
            构建yolo模型结构
        Parameters
        ----------
            inputs:       模型的输入变量
            num_anchors:  每个grid cell负责检测的anchor数量
            num_classes:  类别数量
            training:     是否为训练模式
        """

        conv = features_out[0]
        conv2d_45 = features_out[1]
        conv2d_26 = features_out[2]

        print('conv : ', conv)
        print('conv2d_45 : ', conv2d_45)
        print('conv2d_26 : ', conv2d_26)

        with tf.variable_scope('yolo'):
            conv2d_57, conv2d_59, conv_index = self._yolo_block(conv, filters_yolo_block[0],
                                                                num_anchors * (num_classes + 5),
                                                                conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
            print('conv2d_59 : ', conv2d_59)
            print('conv2d_57 : ', conv2d_57)

            conv2d_60 = self._conv2d_layer(conv2d_57, filters_num=filters_yolo_block[1], kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_60 = self._batch_normalization_layer(conv2d_60, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            print('conv2d_60 : ', conv2d_60)

            conv_index += 1
            upSample_0 = tf.image.resize_nearest_neighbor(conv2d_60,
                                                          [2 * tf.shape(conv2d_60)[1], 2 * tf.shape(conv2d_60)[2]],
                                                          name='upSample_0')
            # upSample_0 = tf.layers.conv2d_transpose(conv2d_60, filters = filters_yolo_block[1], kernel_size = (3, 3), strides = (2, 2), padding = 'same', name='upSample_0')
            print('upSample_0 : ', upSample_0)

            route0 = tf.concat([upSample_0, conv2d_45], axis=-1, name='route_0')
            print('route0 : ', route0)

            conv2d_65, conv2d_67, conv_index = self._yolo_block(route0, filters_yolo_block[1],
                                                                num_anchors * (num_classes + 5),
                                                                conv_index=conv_index, training=training,
                                                                norm_decay=self.norm_decay,
                                                                norm_epsilon=self.norm_epsilon)
            print('conv2d_67 : ', conv2d_67)
            print('conv2d_65 : ', conv2d_65)

            conv2d_68 = self._conv2d_layer(conv2d_65, filters_num=filters_yolo_block[2], kernel_size=1, strides=1,
                                           name="conv2d_" + str(conv_index))
            conv2d_68 = self._batch_normalization_layer(conv2d_68, name="batch_normalization_" + str(conv_index),
                                                        training=training, norm_decay=self.norm_decay,
                                                        norm_epsilon=self.norm_epsilon)
            print('conv2d_68 : ', conv2d_68)

            conv_index += 1
            upSample_1 = tf.image.resize_nearest_neighbor(conv2d_68,
                                                          [2 * tf.shape(conv2d_68)[1], 2 * tf.shape(conv2d_68)[2]],
                                                          name='upSample_1')
            # upSample_1 = tf.layers.conv2d_transpose(conv2d_68, filters = filters_yolo_block[2], kernel_size = (3, 3), strides = (2, 2), padding = 'same', name='upSample_1')
            print('upSample_1 : ', upSample_1)

            route1 = tf.concat([upSample_1, conv2d_26], axis=-1, name='route_1')
            print('route1 : ', route1)

            _, conv2d_75, _ = self._yolo_block(route1, filters_yolo_block[2], num_anchors * (num_classes + 5),
                                               conv_index=conv_index, training=training, norm_decay=self.norm_decay,
                                               norm_epsilon=self.norm_epsilon)
            print('conv2d_75 : ', conv2d_75)

        return [conv2d_59, conv2d_67, conv2d_75]
        # return [tf.sigmoid(conv2d_59), tf.sigmoid(conv2d_67), tf.sigmoid(conv2d_75)]

    def _darknet53_cloud(self, training=False, norm_decay=0.997, norm_epsilon=1e-5):
        """
        Introduction
        ------------
            构建yolo3使用的darknet53网络结构
        Parameters
        ----------
            inputs:       模型输入变量
            weights_dict: 预训练权重
            training:     是否为训练
            norm_decay:   在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            conv:       经过52层卷积计算之后的结果, 输入图片为416x416x3，则此时输出的结果shape为13x13x1024
            route1:     返回第26层卷积计算结果52x52x256, 供后续使用
            route2:     返回第43层卷积计算结果26x26x512, 供后续使用
            conv_index: 卷积层计数，方便在加载预训练模型时使用
        """


        with tf.variable_scope('darknet53') as scope:
            # conv = self._conv2d_layer(inputs, filters_num = 32, kernel_size = 3, strides = 1, name = "conv2d_" + str(conv_index))
            # conv = self._batch_normalization_layer(conv, name = "batch_normalization_" + str(conv_index), training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # conv_index += 1
            # conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 64, blocks_num = 1, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 128, blocks_num = 2, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            # conv, conv_index = self._Residual_block(conv, conv_index = conv_index, filters_num = 256, blocks_num = 8, training = training, norm_decay = norm_decay, norm_epsilon = norm_epsilon)
            conv = tf.placeholder(tf.float32, [None, None, None, 256], name='input')
            conv_index = 27
            route1 = conv
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=512, blocks_num=8,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            route2 = conv
            conv, conv_index = self._Residual_block(conv, conv_index=conv_index, filters_num=1024, blocks_num=4,
                                                    training=training, norm_decay=norm_decay, norm_epsilon=norm_epsilon)
            features_out = [conv, route2, route1]
            filters_yolo_block = [512, 256, 128]

        output = self.yolo_inference(features_out, filters_yolo_block, conv_index, len(self.anchors) / 3,
                                self.num_classes, self.is_training)

        if self.is_training:
            self.global_step = tf.Variable(0, trainable=False)
            self.bbox_true_13 = tf.placeholder(shape=[None, None, None, 3, self.num_classes + 5], dtype=tf.float32)
            self.bbox_true_26 = tf.placeholder(shape=[None, None, None, 3, self.num_classes + 5], dtype=tf.float32)
            self.bbox_true_52 = tf.placeholder(shape=[None, None, None, 3, self.num_classes + 5], dtype=tf.float32)
            self.bbox_true = [self.bbox_true_13, self.bbox_true_26, self.bbox_true_52]
            self.loss = self.yolo_loss(output, self.bbox_true, self.anchors, self.num_classes,
                                       self.config.ignore_thresh)

            learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step, 1000, 0.95,
                                                       staircase=True)
            optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
            update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_op = optimizer.minimize(loss=self.loss, global_step=self.global_step)
        else:
            self.image_shape = tf.placeholder(dtype=tf.int32, shape=(2,), name='shapes')
            self.boxes, self.scores, self.classes = self.yolo_eval(output, self.image_shape, self.config.max_boxes)
            return self.boxes, self.scores, self.classes

    def _yolo_block(self, inputs, filters_num, out_filters, conv_index, training=True, norm_decay=0.997,
                    norm_epsilon=1e-5):
        """
        Introduction
        ------------
            yolo3在Darknet53提取的特征层基础上，又加了针对3种不同比例的feature map的block，这样来提高对小物体的检测率
        Parameters
        ----------
            inputs: 输入特征
            filters_num: 卷积核数量
            out_filters: 最后输出层的卷积核数量
            conv_index: 卷积层数序号，方便根据名字加载预训练权重
            training: 是否为训练
            norm_decay: 在预测时计算moving average时的衰减率
            norm_epsilon: 方差加上极小的数，防止除以0的情况
        Returns
        -------
            route: 返回最后一层卷积的前一层结果
            conv: 返回最后一层卷积的结果
            conv_index: conv层计数
        """
        conv = self._conv2d_layer(inputs, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=filters_num, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        route = conv
        conv = self._conv2d_layer(conv, filters_num=filters_num * 2, kernel_size=3, strides=1,
                                  name="conv2d_" + str(conv_index))
        conv = self._batch_normalization_layer(conv, name="batch_normalization_" + str(conv_index), training=training,
                                               norm_decay=norm_decay, norm_epsilon=norm_epsilon)
        conv_index += 1
        conv = self._conv2d_layer(conv, filters_num=out_filters, kernel_size=1, strides=1,
                                  name="conv2d_" + str(conv_index), use_bias=True)
        conv_index += 1
        return route, conv, conv_index

    def yolo_head(self, feats, anchors, num_classes, input_shape, training=True):
        """
        Introduction
        ------------
            根据不同大小的feature map做多尺度的检测，三种feature map大小分别为13x13x1024, 26x26x512, 52x52x256
        Parameters
        ----------
            feats: 输入的特征feature map
            anchors: 针对不同大小的feature map的anchor
            num_classes: 类别的数量
            input_shape: 图像的输入大小，一般为416
            trainging: 是否训练，用来控制返回不同的值
        Returns
        -------
        """
        print('feats : ', feats)
        print('anchors : ', anchors)
        print('input_shape : ', input_shape)

        num_anchors = len(anchors)
        anchors_tensor = tf.reshape(tf.constant(anchors, dtype=tf.float32), [1, 1, 1, num_anchors, 2])
        grid_size = tf.shape(feats)[1:3]
        predictions = tf.reshape(feats, [-1, grid_size[0], grid_size[1], num_anchors, num_classes + 5])
        # 这里构建13*13*1*2的矩阵，对应每个格子加上对应的坐标
        grid_y = tf.tile(tf.reshape(tf.range(grid_size[0]), [-1, 1, 1, 1]), [1, grid_size[1], 1, 1])
        grid_x = tf.tile(tf.reshape(tf.range(grid_size[1]), [1, -1, 1, 1]), [grid_size[0], 1, 1, 1])
        grid = tf.concat([grid_x, grid_y], axis=-1)
        grid = tf.cast(grid, tf.float32)
        # 将x,y坐标归一化为占416的比例
        box_xy = (tf.sigmoid(predictions[..., :2]) + grid) / tf.cast(grid_size[::-1], tf.float32)
        # 将w,h也归一化为占416的比例
        box_wh = tf.exp(predictions[..., 2:4]) * anchors_tensor / input_shape[::-1]
        box_confidence = tf.sigmoid(predictions[..., 4:5])
        box_class_probs = tf.sigmoid(predictions[..., 5:])
        if training == True:
            return grid, predictions, box_xy, box_wh
        return box_xy, box_wh, box_confidence, box_class_probs

    def yolo_boxes_scores(self, feats, anchors, num_classes, input_shape, image_shape):
        """
        Introduction
        ------------
            该函数是将box的坐标修正，除去之前按照长宽比缩放填充的部分，最后将box的坐标还原成相对原始图片的
        Parameters
        ----------
            feats: 模型输出feature map
            anchors: 模型anchors
            num_classes: 数据集类别数
            input_shape: 训练输入图片大小
            image_shape: 原始图片的大小
        """
        input_shape = tf.cast(input_shape, tf.float32)
        image_shape = tf.cast(image_shape, tf.float32)
        box_xy, box_wh, box_confidence, box_class_probs = self.yolo_head(feats, anchors, num_classes, input_shape,
                                                                         training=False)
        box_yx = box_xy[..., ::-1]
        box_hw = box_wh[..., ::-1]
        new_shape = tf.round(image_shape * tf.reduce_min(input_shape / image_shape))
        offset = (input_shape - new_shape) / 2. / input_shape  # 0 #
        scale = input_shape / new_shape
        box_yx = (box_yx - offset) * scale
        box_hw = box_hw * scale

        box_min = box_yx - box_hw / 2.
        box_max = box_yx + box_hw / 2.
        boxes = tf.concat(
            [box_min[..., 0:1],
             box_min[..., 1:2],
             box_max[..., 0:1],
             box_max[..., 1:2]],
            axis=-1
        )
        boxes *= tf.concat([image_shape, image_shape], axis=-1)
        boxes = tf.reshape(boxes, [-1, 4])
        boxes_scores = box_confidence * box_class_probs
        boxes_scores = tf.reshape(boxes_scores, [-1, num_classes])
        return boxes, boxes_scores

    def box_iou(self, box1, box2):
        """
        Introduction
        ------------
            计算box tensor之间的iou
        Parameters
        ----------
            box1: shape=[grid_size, grid_size, anchors, xywh]
            box2: shape=[box_num, xywh]
        Returns
        -------
            iou:
        """
        box1 = tf.expand_dims(box1, -2)
        box1_xy = box1[..., :2]
        box1_wh = box1[..., 2:4]
        box1_mins = box1_xy - box1_wh / 2.
        box1_maxs = box1_xy + box1_wh / 2.

        box2 = tf.expand_dims(box2, 0)
        box2_xy = box2[..., :2]
        box2_wh = box2[..., 2:4]
        box2_mins = box2_xy - box2_wh / 2.
        box2_maxs = box2_xy + box2_wh / 2.

        intersect_mins = tf.maximum(box1_mins, box2_mins)
        intersect_maxs = tf.minimum(box1_maxs, box2_maxs)
        intersect_wh = tf.maximum(intersect_maxs - intersect_mins, 0.)
        intersect_area = intersect_wh[..., 0] * intersect_wh[..., 1]
        box1_area = box1_wh[..., 0] * box1_wh[..., 1]
        box2_area = box2_wh[..., 0] * box2_wh[..., 1]
        iou = intersect_area / (box1_area + box2_area - intersect_area)
        return iou

    def yolo_loss(self, yolo_output, y_true, anchors, num_classes, ignore_thresh=.5):
        """
        Introduction
        ------------
            yolo模型的损失函数
        Parameters
        ----------
            yolo_output: yolo模型的输出
            y_true: 经过预处理的真实标签，shape为[batch, grid_size, grid_size, 5 + num_classes]
            anchors: yolo模型对应的anchors
            num_classes: 类别数量
            ignore_thresh: 小于该阈值的box我们认为没有物体
        Returns
        -------
            loss: 每个batch的平均损失值
            accuracy
        """
        loss = 0.0
        anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
        input_shape = tf.shape(yolo_output[0])[1: 3] * 32
        input_shape = tf.cast(input_shape, tf.float32)
        grid_shapes = [tf.cast(tf.shape(yolo_output[l])[1:3], tf.float32) for l in range(3)]
        for index in range(3):
            # 只有负责预测ground truth box的grid对应的为1, 才计算相对应的loss
            # object_mask的shape为[batch_size, grid_size, grid_size, 3, 1]
            object_mask = y_true[index][..., 4:5]
            class_probs = y_true[index][..., 5:]
            grid, predictions, pred_xy, pred_wh = self.yolo_head(yolo_output[index], anchors[anchor_mask[index]],
                                                                 num_classes, input_shape, training=True)
            # pred_box的shape为[batch, box_num, 4]
            pred_box = tf.concat([pred_xy, pred_wh], axis=-1)
            raw_true_xy = y_true[index][..., :2] * grid_shapes[index][::-1] - grid
            object_mask_bool = tf.cast(object_mask, dtype=tf.bool)
            raw_true_wh = tf.log(
                tf.where(tf.equal(y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1], 0),
                         tf.ones_like(y_true[index][..., 2:4]),
                         y_true[index][..., 2:4] / anchors[anchor_mask[index]] * input_shape[::-1]))
            # 该系数是用来调整box坐标loss的系数
            box_loss_scale = 2 - y_true[index][..., 2:3] * y_true[index][..., 3:4]
            ignore_mask = tf.TensorArray(dtype=tf.float32, size=1, dynamic_size=True)

            def loop_body(internal_index, ignore_mask):
                # true_box的shape为[box_num, 4]
                true_box = tf.boolean_mask(y_true[index][internal_index, ..., 0:4],
                                           object_mask_bool[internal_index, ..., 0])
                iou = self.box_iou(pred_box[internal_index], true_box)
                # 计算每个true_box对应的预测的iou最大的box
                best_iou = tf.reduce_max(iou, axis=-1)
                ignore_mask = ignore_mask.write(internal_index, tf.cast(best_iou < ignore_thresh, tf.float32))
                return internal_index + 1, ignore_mask

            _, ignore_mask = tf.while_loop(
                lambda internal_index, ignore_mask: internal_index < tf.shape(yolo_output[0])[0], loop_body,
                [0, ignore_mask])
            ignore_mask = ignore_mask.stack()
            ignore_mask = tf.expand_dims(ignore_mask, axis=-1)
            # 计算四个部分的loss
            xy_loss = object_mask * box_loss_scale * tf.nn.sigmoid_cross_entropy_with_logits(labels=raw_true_xy,
                                                                                             logits=predictions[...,
                                                                                                    0:2])
            wh_loss = object_mask * box_loss_scale * 0.5 * tf.square(raw_true_wh - predictions[..., 2:4])
            confidence_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                    logits=predictions[..., 4:5]) + (
                                          1 - object_mask) * tf.nn.sigmoid_cross_entropy_with_logits(labels=object_mask,
                                                                                                     logits=predictions[
                                                                                                            ...,
                                                                                                            4:5]) * ignore_mask
            class_loss = object_mask * tf.nn.sigmoid_cross_entropy_with_logits(labels=class_probs,
                                                                               logits=predictions[..., 5:])
            xy_loss = tf.reduce_sum(xy_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            wh_loss = tf.reduce_sum(wh_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            confidence_loss = tf.reduce_sum(confidence_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)
            class_loss = tf.reduce_sum(class_loss) / tf.cast(tf.shape(yolo_output[0])[0], tf.float32)

            loss += xy_loss + wh_loss + confidence_loss + class_loss

        return loss

    def yolo_eval(self, yolo_outputs, image_shape, max_boxes=20):
        """
        Introduction
        ------------
            根据Yolo模型的输出进行非极大值抑制，获取最后的物体检测框和物体检测类别
        Parameters
        ----------
            yolo_outputs: yolo模型输出
            image_shape: 图片的大小
            max_boxes:  最大box数量
        Returns
        -------
            boxes_: 物体框的位置
            scores_: 物体类别的概率
            classes_: 物体类别
        """
        with tf.variable_scope('boxes_scores'):
            anchor_mask = [[6, 7, 8], [3, 4, 5], [0, 1, 2]]
            boxes = []
            box_scores = []
            input_shape = tf.shape(yolo_outputs[0])[1: 3] * 32
            # 对三个尺度的输出获取每个预测box坐标和box的分数，score计算为置信度x类别概率
            for i in range(len(yolo_outputs)):
                _boxes, _box_scores = self.yolo_boxes_scores(yolo_outputs[i], self.anchors[anchor_mask[i]],
                                                             len(self.class_names), input_shape, image_shape)
                boxes.append(_boxes)
                box_scores.append(_box_scores)
            boxes = tf.concat(boxes, axis=0)
            box_scores = tf.concat(box_scores, axis=0)

        with tf.variable_scope('nms'):
            mask = box_scores >= self.obj_threshold
            max_boxes_tensor = tf.constant(max_boxes, dtype=tf.int32)
            boxes_ = []
            scores_ = []
            classes_ = []
            for c in range(len(self.class_names)):
                class_boxes = tf.boolean_mask(boxes, mask[:, c])
                class_box_scores = tf.boolean_mask(box_scores[:, c], mask[:, c])
                nms_index = tf.image.non_max_suppression(class_boxes, class_box_scores, max_boxes_tensor,
                                                         iou_threshold=self.nms_threshold)
                class_boxes = tf.gather(class_boxes, nms_index)
                class_box_scores = tf.gather(class_box_scores, nms_index)
                classes = tf.ones_like(class_box_scores, 'int32') * c
                boxes_.append(class_boxes)
                scores_.append(class_box_scores)
                classes_.append(classes)

        with tf.variable_scope('output'):
            boxes_ = tf.concat(boxes_, axis=0, name='boxes')
            scores_ = tf.concat(scores_, axis=0, name='scores')
            classes_ = tf.concat(classes_, axis=0, name='classes')
        return boxes_, scores_, classes_

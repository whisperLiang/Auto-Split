# -*- coding: utf-8 -*-


"""
paramters for training
"""

import os

class YoloConfig:

    gpu_index = "0"
    
    net_type = 'darknet53'
    input_shape = (288, 512)
    classes = ['person', 'face']

    log_path = './yolo.log'
    model_dir = './models/%s_%dx%d_%s' % (net_type, input_shape[0], input_shape[1], '&'.join(classes))

    load_imagenet_weights = False
    # imagenet_weights_path = os.path.join(model_dir, '..', 'resnet18.npz')
    # model_checkpoint_path = os.path.join(model_dir, 'yolo3-epoch000-val_loss19.37.ckpt')
    anchors = [10,13,  16,30,  33,23,  30,61,  62,45,  59,119,  116,90,  156,198,  210,240]
    
    data_path = '/home/data_160/data3/smart_home/human_detection/coco'
    train_file = './data/train_Person+Face-coco-20190118.txt'
    val_file = './data/val_Person+Face-coco-20190118.txt'
    
    max_boxes = 50

    norm_decay = 0.99
    norm_epsilon = 1e-5
    ignore_thresh = 0.5
    
    learning_rate = 1e-3
    batch_size = 4
    epoch = 50
    
    obj_threshold = 0.3
    nms_threshold = 0.4
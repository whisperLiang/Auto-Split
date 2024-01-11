
import copy
import math
import time
import os
import logging
from collections import OrderedDict
import numpy as np
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
import torchnet.meter as tnt
import parser
from functools import partial
import argparse

import torch
import argparse
from collections import OrderedDict
from torch.jit.annotations import Tuple, List, Dict, Optional
from copy import deepcopy

# Distiller imports
import distiller.quantization.ptq_bit_search as lapq_bit_search
import distiller.quantization.ptq_coordinate_search as lapq
import distiller
import distiller.apputils as apputils
from distiller.data_loggers import *
import distiller.quantization as quantization
import distiller.models as models
from distiller.models import create_model
from distiller.utils import float_range_argparse_checker as float_range

# obj_detection imports
from obj_detection.pq_detection.compress_detector import get_dataset,get_transform,utils,patch_fastrcnn
from obj_detection.models import resnet
from obj_detection.pq_detection.engine import train_one_epoch, evaluate
from obj_detection.models.split_dnn_utils import compare_params_with_subset_dnn, copy_params_from_parent_dnn
from obj_detection.models.faster_rcnn import split_fasterrcnn_resnet50_fpn
from obj_detection.pq_detection.ptq_compress import EdgeDNN
# torchvision imports
from torchvision.models.detection.transform import GeneralizedRCNNTransform
import torchvision.models.detection as detection
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN

# Logger handle
msglogger = logging.getLogger()

def init_detection_compression_arg_parser(include_ptq_lapq_args=False):
    '''Object detection compression application command-line arguments.
    '''
    QUANT_CHOICES = ['bit_search', 'fake_pq', 'pq', 'eval']
    parser = argparse.ArgumentParser(description='Distiller image classification model compression')

    parser.add_argument('--data-path', default='~/datasets/coco2017', help='dataset')
    # parser.add_argument('data', metavar='DATASET_DIR', help='path to dataset', default='~/datasets/coco2017')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--quant-method', type=lambda s: s.lower(), choices=QUANT_CHOICES, default='bit_search',
                        help='print a summary of the model, and exit - options: | '.join(QUANT_CHOICES))

    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--arch', '-a', metavar='ARCH', default='fasterrcnn_resnet50_fpn', type=lambda s: s.lower(),
                        choices=models.ALL_MODEL_NAMES,
                        help='model architecture: ' +
                        ' | '.join(models.ALL_MODEL_NAMES) +
                        ' (default: fasterrcnn_resnet50_fpn)')

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='starting epoch number')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')

    #--------------------------------------------------------------------------------------------------
    # -- Optimizer args

    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')

    optimizer_args = parser.add_argument_group('Optimizer arguments')
    optimizer_args.add_argument('--lr', '--learning-rate', default=0.1,
                    type=float, metavar='LR', help='initial learning rate')
    optimizer_args.add_argument('--momentum', default=0.9, type=float,
                    metavar='M', help='momentum')
    optimizer_args.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

    #--------------------------------------------------------------------------------------------------
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)

    parser.add_argument('--print-freq', '-p', default=10, type=int,
                        metavar='N', help='print frequency (default: 10)')
    parser.add_argument('--verbose', '-v', action='store_true', help='Emit debug log messages')
    #--------------------------------------------------------------------------------------------------


    parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                        help='Only test the model')
    parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                        help='collect activation statistics on phases: train, valid, and/or test'
                        ' (WARNING: this slows down training)')

    parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                        help='configuration file for pruning the model (default is to use hard-coded schedule)')

    parser.add_argument('--gpus', metavar='DEV_ID', default=None,
                        help='Comma-separated list of GPU device IDs to be used '
                             '(default is to use all available devices)')

    parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    distiller.quantization.add_post_train_quant_args(parser, add_lapq_args=include_ptq_lapq_args)
    return parser

def load_test_data():
    dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)
    test_sampler = torch.utils.data.SequentialSampler(dataset_test)
    test_loader = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    return test_loader, num_classes

def transform_images(images, targets=None, is_training=False):
    """
    Arguments:
        images (list[Tensor]): images to be processed
        targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)
    """
    if is_training and targets is None:
        raise ValueError("In training mode, targets should be passed")

    min_size = 800
    max_size = 1333
    image_mean = None
    image_std = None

    if image_mean is None:
        image_mean = [0.485, 0.456, 0.406]
    if image_std is None:
        image_std = [0.229, 0.224, 0.225]

    transform = GeneralizedRCNNTransform(min_size, max_size, image_mean, image_std)
    original_image_sizes = torch.jit.annotate(List[Tuple[int, int]], [])
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    images, targets = transform(images, targets)

    return images, targets

# TODO: repeated forloops over once over _get_dummy_input & then in transform images. Optimize
def get_dummy_input_from_dataset(data_loader):
    num_images = 1
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # returns the first image. Clean up
    for image, targets in metric_logger.log_every(data_loader, num_images, header):
        image = list(img.to(device) for img in image)
        image, targets = transform_images(image)
        return image.tensors


if __name__ == '__main__':

    parser = init_detection_compression_arg_parser(include_ptq_lapq_args=True)
    args = parser.parse_args()


    device = torch.device(args.device)
    script_dir = os.path.dirname(__file__)
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
    if utils.is_main_process():
        msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir,
                                             args.verbose)

        # Log various details about the execution environment.  It is sometimes useful
        # to refer to past experiment executions and this information may be useful.
        apputils.log_execution_env_state(
            filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
            msglogger.logdir)
        msglogger.debug("Distiller: %s", distiller.__version__)
    else:
        msglogger = logging.getLogger()
        msglogger.disabled = True

    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    # Load data
    test_data_loader, num_classes = load_test_data()

    msglogger.info("Creating model")
    orig_detection_model = detection.__dict__[args.model](num_classes=num_classes,
                                                          pretrained=True)
    patch_fastrcnn(orig_detection_model)
    orig_detection_model.to(device)

    edge_dnn = EdgeDNN(test_data_loader, num_classes, criterion, args)
    quantizer_model,_ = edge_dnn.ptq_edge_dnn()
    quantizer_model.to(device)

    split_detection_model = split_fasterrcnn_resnet50_fpn(edge_dnn=quantizer_model,edge_dnn_len=72, num_classes= num_classes,
                                                          pretrained=True)

    split_detection_model.to(device)
    split_detection_model.eval()
    x = [torch.rand(3, 300, 400), torch.rand(3, 500, 400)]
    x = [i.to(device) for i in x ]
    predictions = split_detection_model(x)
    print('')


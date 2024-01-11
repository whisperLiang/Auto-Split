#
# Copyright (c) 2019 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

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

# torchvision imports
from torchvision.models.detection.transform import GeneralizedRCNNTransform
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
import torchvision.models.detection as torch_detection

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

    parser.add_argument('--print-freq', '-p', default=1, type=int,
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
def get_dummy_input_from_dataset(data_loader, device):
    num_images = 1
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'
    # returns the first image. Clean up
    for image, targets in metric_logger.log_every(data_loader, num_images, header):
        image = list(img.to(device) for img in image)
        image, targets = transform_images(image)
        return image.tensors

# edge_model = resnet.resnet_edge()
class EdgeDNN():
    def __init__(self, edge_model, test_data_loader, num_classes, criterion, args):
        self.criterion = criterion
        self.args = deepcopy(args)
        self.test_data_loader = test_data_loader
        self.num_classes = num_classes

        msglogger.info("Creating Edge model")
        self.edge_dnn = edge_model
        self.copy_params_detection(num_classes)
        # move to cuda
        self.edge_dnn.to(args.device)
        self.edge_dnn.eval()

    def copy_params_detection(self, num_classes):
        print("Creating Original torch model")
        orig_faster_rcnn =  torch_detection.__dict__['fasterrcnn_resnet50_fpn'](num_classes=num_classes,
                                                              pretrained=True)
        pretrained_dict = orig_faster_rcnn.backbone.body.state_dict()
        model_dict = self.edge_dnn.state_dict()
        # 1. filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        # 2. overwrite entries in the existing state dict
        model_dict.update(pretrained_dict)
        # 3. load the new state dict
        self.edge_dnn.load_state_dict(pretrained_dict)

    def ptq_edge_dnn(self, is_eval=False):
        model = self.edge_dnn
        criterion = self.criterion
        test_data_loader = self.test_data_loader
        args = self.args

        device = next(model.parameters()).device
        test_fn = partial(self.evaluate_edge, data_loader=test_data_loader, device=device, print_freq=args.print_freq)

        dummy_input = get_dummy_input_from_dataset(test_data_loader, device)
        if args.quant_method == 'bit_search':
            msglogger.info('Bit Search: evaluating Quantized Edge DNN with stat collection  ')
            quantizer = distiller.quantization.PostTrainLinearQuantizerBitSearch.from_args(model, args)
            quantizer.model, features = lapq_bit_search.ptq_bit_search(quantizer, dummy_input, test_fn, test_fn=test_fn,
                                                        **lapq.cmdline_args_to_dict(args), is_eval=is_eval)
            return quantizer.model, features
        elif args.quant_method == 'fake_pq':
            msglogger.info('PTQ Fake Quantization: evaluating Quantized Edge DNN')
            quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(model, args)
            model, features = lapq.ptq_fake_quantization(quantizer, dummy_input, test_fn, test_fn=test_fn,
                                                        **lapq.cmdline_args_to_dict(args), is_eval=is_eval)
            return model, features
        elif args.quant_method == 'pq':
            msglogger.info('PTQ: evaluating Quantized Edge DNN')
            quantizer, features = self.quantize_detection_model(model, criterion, test_fn, args,
                                                       test_data_loader, loggers=None, save_flag=True, is_eval=is_eval)
            return quantizer.model, features
        elif args.quant_method == 'eval':
            msglogger.info('evaluating Edge Float model (not quantized)')
            model.to(args.device)
            if is_eval:
                features = test_fn(model)
            else:
                features = None
            return model, features
        else:
            raise ValueError('wrong args.quant_method .. Given {}'.format(args.quant_method))

    def quantize_detection_model(self, model, criterion, test_fn, args, test_loader, loggers=None, save_flag=True, is_eval=False):
        """Collect stats using test_loader (when stats file is absent),

        clone the model and quantize the clone, and finally, test it.
        args.device is allowed to differ from the model's device.
        When args.qe_calibration is set to None, uses 0.05 instead.

        scheduler - pass scheduler to store it in checkpoint
        save_flag - defaults to save both quantization statistics and checkpoint.
        """
        if hasattr(model, 'quantizer_metadata') and \
                model.quantizer_metadata['type'] == distiller.quantization.PostTrainLinearQuantizer:
            raise RuntimeError('Trying to invoke post-training quantization on a model that has already been post-'
                               'train quantized. Model was likely loaded from a checkpoint. Please run again without '
                               'passing the --quantize-eval flag')

        if not (args.qe_dynamic or args.qe_stats_file or args.qe_config_file):
            args_copy = copy.deepcopy(args)
            args_copy.qe_calibration = args.qe_calibration if args.qe_calibration is not None else 0.05

            # set stats into args stats field
            args.qe_stats_file = self.acts_quant_stats_collection(
                model, criterion, loggers, args_copy, test_loader=test_loader, save_to_file=save_flag)

        args_qe = copy.deepcopy(args)
        if args.device == 'cpu':
            # NOTE: Even though args.device is CPU, we allow here that model is not in CPU.
            qe_model = distiller.make_non_parallel_copy(model).cpu()
        else:
            qe_model = copy.deepcopy(model).to(args.device)

        quantizer = quantization.PostTrainLinearQuantizer.from_args(qe_model, args_qe)
        dummy_input = get_dummy_input_from_dataset(test_loader, args.device)
        quantizer.prepare_model(dummy_input)

        # if args.qe_convert_pytorch:
        #     qe_model = _convert_ptq_to_pytorch(qe_model, args_qe)

        if is_eval:
            features = test_fn(model)
        else:
            features = None
        # if save_flag:
        #     checkpoint_name = 'quantized'
        #     apputils.save_checkpoint(0, args_qe.arch, qe_model, scheduler=scheduler,
        #         name='_'.join([args_qe.name, checkpoint_name]) if args_qe.name else checkpoint_name,
        #         dir=msglogger.logdir, extras={'quantized_top1': test_res[0]})
        #
        # del qe_model
        return quantizer, features

    def acts_quant_stats_collection(self, model, criterion, loggers, args, test_loader, save_to_file=False):
        msglogger.info('Collecting quantization calibration stats based on {:.1%} of test dataset'
                       .format(args.qe_calibration))
        # if test_loader is None:
        #     tmp_args = copy.deepcopy(args)
        #     tmp_args.effective_test_size = tmp_args.qe_calibration
        #     # Batch size 256 causes out-of-memory errors on some models (due to extra space taken by
        #     # stats calculations). Limiting to 128 for now.
        #     # TODO: Come up with "smarter" limitation?
        #     tmp_args.batch_size = min(128, tmp_args.batch_size)
        #     test_loader = load_data(tmp_args, fixed_subset=True, load_train=False, load_val=False)
        # test_fn = partial(test, test_loader=test_loader, criterion=criterion,
        #                   loggers=loggers, args=args, activations_collectors=None)

        # TODO: add orig resnet50 layers w/o quantized to compare if required.
        test_fn = partial(self.evaluate_edge, data_loader=test_loader, device=args.device, print_freq=args.print_freq)

        with distiller.get_nonparallel_clone_model(model) as cmodel:
            return collect_quant_stats(cmodel, test_fn, classes=None,
                                       inplace_runtime_check=True, disable_inplace_attrs=True,
                                       save_dir=msglogger.logdir if save_to_file else None)

    def evaluate_edge(self, model, data_loader, device, print_freq):
        # n_threads = torch.get_num_threads()
        # FIXME remove this and make paste_masks_in_image run on the GPU
        # torch.set_num_threads(1)
        cpu_device = torch.device("cpu")
        model.eval()
        model.to(device)
        metric_logger = utils.MetricLogger(delimiter="  ")
        header = 'Test:'

        idx = 0
        # features = []

        for image, targets in metric_logger.log_every(data_loader, print_freq, header):
            image = list(img.to(device) for img in image)
            image, targets = transform_images(image)

            # for x in image:
            # print(image.image_sizes)
            outputs = self._evaluate_edge(model, image)
            # features.append(outputs)
            idx += 1

        # return features

    def _evaluate_edge(self, model, images):
        features = model(images.tensors)

        # msglogger.info('len images {}'.format(type(images)))
        # msglogger.info('features shape: {}'.format(len(features)))
        if isinstance(features, torch.Tensor):
            features = OrderedDict([('0', features)])

        return features

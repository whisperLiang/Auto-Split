# This code is originally from:
#   https://github.com/pytorch/vision/tree/v0.4.2/references/detection/train.py
# It contains code to support compression (distiller)
r"""PyTorch Detection Training.

To run in a multi-gpu environment, use the distributed launcher::

    python -m torch.distributed.launch --nproc_per_node=$NGPU --use_env \
        compress_detector.py ... --world-size $NGPU

"""
import datetime
import os
import time

import torch
import torch.utils.data
from torch import nn
import torchvision
import torchvision.models.detection as detection
from torchvision.models.detection.generalized_rcnn import GeneralizedRCNN
from torchvision.ops.misc import FrozenBatchNorm2d
import torch.distributed as dist

import distiller
from distiller.data_loggers import *
import distiller.apputils as apputils
import distiller.pruning
import distiller.models
from distiller.model_transforms import fold_batch_norms
from obj_detection.pq_detection.coco_utils import get_coco, get_coco_kp

from obj_detection.pq_detection.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from obj_detection.pq_detection.engine import train_one_epoch, evaluate

import  obj_detection.pq_detection.utils as utils
from obj_detection.pq_detection.compress_detector import get_dataset, get_transform, patch_fastrcnn

import logging
logging.getLogger().setLevel(logging.INFO)  # Allow distiller info to be logged.


def get_dataloader(args, script_dir):

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

    # Data loading code
    print("Loading data")
    dataset_test, num_classes = get_dataset(args.dataset, "val", get_transform(train=False), args.data_path)

    print("Creating data loaders")
    if args.distributed:
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)


    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    return  data_loader_test, num_classes ,msglogger

def main(args):
    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))
    data_loader_test, num_classes, msglogger = get_dataloader(args, script_dir)

    print("Creating model")
    model = detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained)
    patch_fastrcnn(model)
    model.to(device)

    model_without_ddp = model

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(
        params, lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step_size, gamma=args.lr_gamma)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.lr_steps, gamma=args.lr_gamma)

    compression_scheduler = None
    if utils.is_main_process():
        # Create a couple of logging backends.  TensorBoardLogger writes log files in a format
        # that can be read by Google's Tensor Board.  PythonLogger writes to the Python logger.
        tflogger = TensorBoardLogger(msglogger.logdir)
        pylogger = PythonLogger(msglogger)

    # if args.compress:
    #     # The main use-case for this sample application is CNN compression. Compression
    #     # requires a compression schedule configuration file in YAML.
    #     compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler, None)
    #     # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
    #     model.to(args.device)
    # elif compression_scheduler is None:
    #     compression_scheduler = distiller.CompressionScheduler(model)

    if args.test_only:
        evaluate(model, data_loader_test, device=device)
        return


def add_distiller_compression_args(parser):
    SUMMARY_CHOICES = ['sparsity', 'model', 'modules']
    distiller_parser = parser.add_argument_group('Distiller related arguemnts')
    distiller_parser.add_argument('--summary', type=lambda s: s.lower(), choices=SUMMARY_CHOICES, action='append',
                        help='print a summary of the model, and exit - options: | '.join(SUMMARY_CHOICES))
    distiller_parser.add_argument('--export-onnx', action='store', nargs='?', type=str, const='model.onnx',
                                  default=None,
                                  help='export model to ONNX format')
    distiller_parser.add_argument('--compress', dest='compress', type=str, nargs='?', action='store',
                                  help='configuration file for pruning the model '
                                       '(default is to use hard-coded schedule)')
    distiller.pruning.greedy_filter_pruning.add_greedy_pruner_args(distiller_parser)
    distiller_parser.add_argument('--name', '-n', metavar='NAME', default=None, help='Experiment name')
    distiller_parser.add_argument('--verbose', '-v', action='store_true', help='Emit debug log messages')
    distiller.quantization.add_post_train_quant_args(distiller_parser, add_lapq_args=True)
    distiller_parser.add_argument('--activation-stats', '--act-stats', nargs='+', metavar='PHASE', default=list(),
                                  help='collect activation statistics on phases: train, valid, and/or test'
                                  ' (WARNING: this slows down training)')
    distiller_parser.add_argument('--masks-sparsity', dest='masks_sparsity', action='store_true', default=False,
                                  help='print masks sparsity table at end of each epoch')


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='~/datasets/coco2017', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    # parser.add_argument('--epochs', default=1, type=int, metavar='N',
    #                     help='number of total epochs to run')
    # parser.add_argument('--start-epoch', default=0, type=int, help='starting epoch number')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    # parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    # parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
    #                     help='momentum')
    # parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
    #                     metavar='W', help='weight decay (default: 1e-4)',
    #                     dest='weight_decay')
    # parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    # parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    # parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
    parser.add_argument('--print-freq', default=20, type=int, help='print frequency')
    parser.add_argument('--output-dir', default='.', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--aspect-ratio-group-factor', default=0, type=int)
    parser.add_argument(
        "--evaluate",
        dest="test_only",
        help="Only test the model",
        action="store_true",
    )
    parser.add_argument(
        "--pretrained",
        dest="pretrained",
        help="Use pre-trained models from the modelzoo",
        action="store_true",
    )

    # distributed training parameters
    # parser.add_argument('--world-size', default=1, type=int,
    #                     help='number of distributed processes')
    # parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    add_distiller_compression_args(parser)

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)

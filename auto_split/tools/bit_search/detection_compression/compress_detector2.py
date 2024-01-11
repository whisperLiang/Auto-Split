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

from tools.bit_search.detection_compression.coco_utils import get_coco, get_coco_kp
from tools.bit_search.detection_compression.group_by_aspect_ratio import GroupedBatchSampler, create_aspect_ratio_groups
from tools.bit_search.detection_compression.engine import train_one_epoch, evaluate
from tools.bit_search.detection_compression.detection_ptq_lapq import detection_ptq_lapq

import tools.bit_search.detection_compression.transforms as T
import tools.bit_search.detection_compression.utils as utils



import logging
logging.getLogger().setLevel(logging.INFO)  # Allow distiller info to be logged.


def get_dataset(name, image_set, transform, data_path, effective_test_size=None):
    paths = {
        "coco": (data_path, get_coco, 91),
        "coco_kp": (data_path, get_coco_kp, 2)
    }
    p, ds_fn, num_classes = paths[name]

    ds = ds_fn(p, image_set=image_set, transforms=transform, effective_test_size=None)
    return ds, num_classes


def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))
    return T.Compose(transforms)


def patch_fastrcnn(model):
    import torch
    import torchvision

    """
    TODO - complete quantization example
    Partial patch for torchvision's FastRCNN models to allow quantization, by replacing all FrozenBatchNorm2d
    with regular nn.BatchNorm2d-s.
    Args:
        model (GeneralizedRCNN): the model to patch
    """
    assert isinstance(model, GeneralizedRCNN)

    def replace_frozen_bn(frozen_bn: FrozenBatchNorm2d):
        num_features = frozen_bn.weight.shape[0]
        bn = nn.BatchNorm2d(num_features)
        eps = bn.eps
        bn.weight.data = frozen_bn.weight.data
        bn.bias.data = frozen_bn.bias.data
        bn.running_mean.data = frozen_bn.running_mean.data
        bn.running_var.data = frozen_bn.running_var.data
        return bn.eval()

    for n, m in model.named_modules():
        if isinstance(m, FrozenBatchNorm2d):
            distiller.model_setattr(model, n, replace_frozen_bn(m))


def main(args):


    utils.init_distributed_mode(args)
    print(args)

    device = torch.device(args.device)

    script_dir = os.path.dirname(__file__)
    module_path = os.path.abspath(os.path.join(script_dir, '..', '..'))

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
    # dataset_train, num_classes = get_dataset(args.dataset, "train", get_transform(train=True), args.data_path)
    args.effective_test_size=None # Runs all validation set
    effective_test_size_bak = args.effective_test_size
    args.effective_test_size = args.lapq_eval_size
    dataset_eval, num_classes = get_dataset(args.dataset, "val", get_transform(train=False),
                                            args.data_path, args.effective_test_size)
    args.effective_test_size = effective_test_size_bak
    #TODO: Make sure eval & test dataset do not overlap.
    dataset_test, _ = get_dataset(args.dataset, "val", get_transform(train=False),
                                  args.data_path, args.effective_test_size)

    print("Creating data loaders")
    if args.distributed:
        eval_sampler = torch.utils.data.distributed.DistributedSampler(dataset_eval)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test)
    else:
        eval_sampler = torch.utils.data.SequentialSampler(dataset_eval)
        test_sampler = torch.utils.data.SequentialSampler(dataset_test)

    data_loader_eval = torch.utils.data.DataLoader(
        dataset_eval, batch_size=1,
        sampler=eval_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=1,
        sampler=test_sampler, num_workers=args.workers,
        collate_fn=utils.collate_fn)

    print("Creating model")
    model = detection.__dict__[args.model](num_classes=num_classes,
                                                              pretrained=args.pretrained)
    patch_fastrcnn(model)
    model.to(device)
    model.eval()


    if args.summary:
        if utils.is_main_process():
            for summary in args.summary:
                distiller.model_summary(model, summary, args.dataset)
        return

    if args.test_only and args.quantize_eval and args.qe_lapq:
            # criterion = nn.CrossEntropyLoss().to(args.device)
            # tflogger = TensorBoardLogger(msglogger.logdir)
            # pylogger = PythonLogger(msglogger)

            detection_ptq_lapq(model, data_loader_eval, data_loader_test, args)
            return

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

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

    if args.compress:
        # The main use-case for this sample application is CNN compression. Compression
        # requires a compression schedule configuration file in YAML.
        compression_scheduler = distiller.file_config(model, optimizer, args.compress, compression_scheduler, None)
        # Model is re-transferred to GPU in case parameters were added (e.g. PACTQuantizer)
        model.to(args.device)
    elif compression_scheduler is None:
        compression_scheduler = distiller.CompressionScheduler(model)

    if args.qe_calibration:
        def test_fn(model):
            return evaluate(model, data_loader_eval, device=device)
        collect_quant_stats(model_without_ddp, test_fn, save_dir=args.output_dir,
                            modules_to_collect=['backbone', 'rpn', 'roi_heads'])
        # We skip `.transform` because it is a pre-processing unit.
        return

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        model_without_ddp.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        if compression_scheduler and 'compression_scheduler' in checkpoint:
            compression_scheduler.load_state_dict(checkpoint['compression_scheduler'])

    if args.test_only:
        evaluate(model, data_loader_eval, device=device)
        return


class missingdict(dict):
    """This is a little trick to prevent KeyError"""
    def __missing__(self, key):
        return None  # note, does *not* set self[key] - we don't want defaultdict's behavior


def create_activation_stats_collectors(model, *phases):
    """Create objects that collect activation statistics.

    This is a utility function that creates two collectors:
    1. Fine-grade sparsity levels of the activations
    2. L1-magnitude of each of the activation channels

    Args:
        model - the model on which we want to collect statistics
        phases - the statistics collection phases: train, valid, and/or test

    WARNING! Enabling activation statsitics collection will significantly slow down training!
    """
    genCollectors = lambda: missingdict({
        "sparsity":      SummaryActivationStatsCollector(model, "sparsity",
                                                         lambda t: 100 * distiller.utils.sparsity(t)),
        "l1_channels":   SummaryActivationStatsCollector(model, "l1_channels",
                                                         distiller.utils.activation_channels_l1),
        "apoz_channels": SummaryActivationStatsCollector(model, "apoz_channels",
                                                         distiller.utils.activation_channels_apoz),
        "mean_channels": SummaryActivationStatsCollector(model, "mean_channels",
                                                         distiller.utils.activation_channels_means),
        "records":       RecordsActivationStatsCollector(model, classes=[torch.nn.Conv2d])
    })

    return {k: (genCollectors() if k in phases else missingdict())
            for k in ('train', 'valid', 'test')}


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

    import torch
    import torchvision

    dummy_input = torch.rand(1, 3, 480, 640)
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True).eval()
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True).eval()
    backbone_script = torch.jit.script(model.backbone)
    trace_0 = torch.jit.trace(backbone_script, dummy_input)

    trace,Bxx = torch.jit._get_trace_graph(model, dummy_input, _force_outplace=True)
    # torch.jit.save(backbone_script, './Fasterrcnn_backbone.pt')
    from collections import OrderedDict, defaultdict
    pre_dropout_nodes_scope_names = OrderedDict()

    prev_non_dropout_op = None
    xx = [x for x in trace_0.graph.nodes()]
    for node in trace.graph.nodes():
        kind = node.kind()
        if 'aten' not in kind:
            continue
        if kind == 'aten::dropout':
            if prev_non_dropout_op:
                pre_dropout_nodes_scope_names[node.scopeName()] = prev_non_dropout_op.scopeName()
        else:
            prev_non_dropout_op = node


    torch.onnx._optimize_trace(trace, torch.onnx.OperatorExportTypes.ONNX)

    graph = trace.graph()
    self_ops = OrderedDict()
    self_module_ops_map = defaultdict(list)
    self_params = OrderedDict()
    self_edges = []
    self_temp = OrderedDict()

    in_out = list(graph.inputs()) + list(graph.outputs())
    for node in graph.nodes():
        print(node)


    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # model.eval()
    # test_data = torch.rand(1, 3, 480, 640)
    # traced_model = torch.jit.trace(model, test_data)


    import argparse
    parser = argparse.ArgumentParser(
        description=__doc__)

    parser.add_argument('--data-path', default='/datasets01/COCO/022719/', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='maskrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cpu', help='device')
    parser.add_argument('-b', '--batch-size', default=2, type=int)
    parser.add_argument('--epochs', default=13, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='starting epoch number')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')
    parser.add_argument('--lr', default=0.02, type=float, help='initial learning rate')
    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='momentum')
    parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                        metavar='W', help='weight decay (default: 1e-4)',
                        dest='weight_decay')
    parser.add_argument('--lr-step-size', default=8, type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-steps', default=[8, 11], nargs='+', type=int, help='decrease lr every step-size epochs')
    parser.add_argument('--lr-gamma', default=0.1, type=float, help='decrease lr by a factor of lr-gamma')
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
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    add_distiller_compression_args(parser)

    args = parser.parse_args()

    if args.output_dir:
        utils.mkdir(args.output_dir)

    main(args)

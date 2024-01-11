# This code is originally from:
#   https://github.com/pytorch/vision/tree/v0.4.2/references/detection
# (old commit)
# It contains code to support compression (distiller)
import math
import sys
import time
import torch
from torch.jit.annotations import Tuple, List, Dict, Optional
import torchvision.models.detection.mask_rcnn

from obj_detection.pq_detection.coco_utils import get_coco_api_from_dataset
from obj_detection.pq_detection.coco_eval import CocoEvaluator
import obj_detection.pq_detection.utils as utils
from collections import OrderedDict

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, compression_scheduler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    steps_per_epoch = len(data_loader)
    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)

    for train_step, (images, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if compression_scheduler:
            compression_scheduler.on_minibatch_begin(epoch, train_step, steps_per_epoch, optimizer)

        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        loss_dict = model(images, targets)

        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)
        if compression_scheduler:
            losses = compression_scheduler.before_backward_pass(epoch, train_step, steps_per_epoch, losses,
                                                                optimizer=optimizer)

        optimizer.zero_grad()
        losses.backward()

        if compression_scheduler:
            compression_scheduler.before_parameter_optimization(epoch, train_step, steps_per_epoch, optimizer)
        optimizer.step()

        if compression_scheduler:
            compression_scheduler.on_minibatch_end(epoch, train_step, steps_per_epoch, optimizer)

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])



def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def forward_backbone_features_orig(model, images, targets=None):
    images, targets = model.transform(images, targets)
    features = model.backbone.body(images.tensors)
    return features

@torch.no_grad()
def forward_backbone_orig(model, images, targets=None):
    images, targets = model.transform(images, targets)
    features = model.backbone(images.tensors)
    return features

@torch.no_grad()
def forward_backbone_edge_orig(model, images, targets=None):

    images, targets = model.transform(images, targets)
    features = model.backbone(images.tensors)
    return features


@torch.no_grad()
def forward_transform_orig(model, images, targets=None):
    # type: (List[Tensor], Optional[List[Dict[str, Tensor]]]) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]
    """
    Arguments:
        images (list[Tensor]): images to be processed
        targets (list[Dict[Tensor]]): ground-truth boxes present in the image (optional)

    Returns:
        result (list[BoxList] or dict[Tensor]): the output from the model.
            During training, it returns a dict[Tensor] which contains the losses.
            During testing, it returns list[BoxList] contains additional fields
            like `scores`, `labels` and `mask` (for Mask R-CNN models).

    """
    images, targets = model.transform(images, targets)
    return images, targets

@torch.no_grad()
def forward_backbone_features(model, images, targets=None):
    images, targets = model.transform(images, targets)
    features = model.backbone.forward_features(images.tensors)
    return features

@torch.no_grad()
def evaluate(model, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    idx = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        # print('-- image # {}'.format(idx))
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        evaluator_time = time.time()
        coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        idx += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    torch.set_num_threads(n_threads)
    return coco_evaluator

# ------------------------------------------------
# Functions to debug various parts of the network
# ------------------------------------------------

@torch.no_grad()
def evaluate_transforms_single_thread(orig_model, split_model, data_loader, device):
    # n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    orig_model.eval()
    split_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    idx = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        # print('-- image # {}'.format(idx))
        orig_images, orig_targets = forward_transform_orig(orig_model, image)
        split_images, split_targets = split_model.forward_transform(image)
        criterion = torch.nn.MSELoss(reduction='sum')
        loss = criterion(orig_images.tensors, split_images.tensors)
        print('loss: {}'.format(loss))
        print(orig_images.tensors.shape)
        print(split_images.tensors.shape)

        evaluator_time = time.time()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        idx += 1

    return

@torch.no_grad()
def evaluate_feature_comparison(orig_model, split_model, data_loader, device):
    cpu_device = torch.device("cpu")
    orig_model.eval()
    split_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    idx = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torch.cuda.synchronize()
        model_time = time.time()
        # print('-- image # {}'.format(idx))
        orig_outputs = forward_backbone_features_orig(orig_model, image)
        split_outputs = split_model.forward_backbone_features(image)

        print('-- split keys --')
        loss = OrderedDict()
        criterion = torch.nn.MSELoss(reduction='sum')
        for key, split_dnn_item in split_outputs.items():

            if key.isnumeric():
                key = int(key)

            # print(type(key), key)
            if key in orig_outputs:
                orig_dnn_item = orig_outputs[key]
                loss[key] = criterion(split_dnn_item,orig_dnn_item)
                print(key, orig_dnn_item.size())
            else:
                raise ValueError('Feature keys do not match')

        print('-- loss --')
        print(loss)

    return

@torch.no_grad()
def evaluate_backbones_single_thread(orig_model, split_model, data_loader, device):
    # n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    # torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    orig_model.eval()
    split_model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    # coco = get_coco_api_from_dataset(data_loader.dataset)
    # iou_types = _get_iou_types(model)
    # coco_evaluator = CocoEvaluator(coco, iou_types)

    idx = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        # torch.cuda.synchronize()
        model_time = time.time()
        # print('-- image # {}'.format(idx))
        orig_outputs = forward_backbone_orig(orig_model, image)
        split_outputs = split_model.forward_backbone(image)
        # print('-- split keys --')
        loss = OrderedDict()
        criterion = torch.nn.MSELoss(reduction='sum')
        for key, split_dnn_item in split_outputs.items():

            if key.isnumeric():
                key = int(key)

            # print(type(key), key)
            if key in orig_outputs:
                orig_dnn_item = orig_outputs[key]
                loss[key] = criterion(split_dnn_item,orig_dnn_item)
            else:
                raise ValueError('Feature keys do not match')

        # print('-- orig keys --')
        # for key, item in orig_outputs.items():
        #     print(type(key), key)
        #
        # print('-- orig_outputs --')
        # criterion = torch.nn.MSELoss(reduction='sum')
        # loss = []
        #
        # for key, item1 in orig_outputs.items():
        #     print(key)
        #     if key in split_outputs:
        #         item2 = split_outputs[key]
        #         loss.append(criterion(item1,item2))
        #
        #

        print('-- loss --')
        print(loss)

        # # print(orig_outputs)
        # for key, item in orig_outputs.items():
        #     print(key,item.shape)
        #
        # print('-- split_outputs --')
        # for key, item in split_outputs.items():
        #     print(key,item.shape)
        # # print(split_outputs)

        # orig_outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in orig_outputs]
        # split_outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in split_outputs]
        model_time = time.time() - model_time

        # orig_res = {target["image_id"].item(): output for target, output in zip(targets, orig_outputs)}
        # split_res = {target["image_id"].item(): output for target, output in zip(targets, split_outputs)}
        #
        # print('-- orig_results --')
        # print(orig_res)
        #
        # print('-- split_results --')
        # print(split_res)

        evaluator_time = time.time()
        # coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        idx += 1

    # gather the stats from all processes
    # metric_logger.synchronize_between_processes()
    # print("Averaged stats:", metric_logger)
    # coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    # coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    # torch.set_num_threads(n_threads)
    # return coco_evaluator
    return


@torch.no_grad()
def evaluate_compare(model1, model2, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model1.eval()
    model2.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types1 = _get_iou_types(model1)
    coco_evaluator1 = CocoEvaluator(coco, iou_types1)

    iou_types2 = _get_iou_types(model2)
    coco_evaluator2 = CocoEvaluator(coco, iou_types2)

    idx = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        # print('-- image # {}'.format(idx))
        outputs1 = model1(image)
        outputs2 = model2(image)

        outputs1 = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs1]
        outputs2 = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs2]

        model_time = time.time() - model_time

        res1 = {target["image_id"].item(): output for target, output in zip(targets, outputs1)}
        res2 = {target["image_id"].item(): output for target, output in zip(targets, outputs2)}
        # print('res1: {}'.format(res1))
        # print('res2: {}'.format(res2))
        evaluator_time = time.time()

        coco_evaluator1.update(res1)
        coco_evaluator2.update(res2)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
        idx += 1

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats1:", metric_logger)
    coco_evaluator1.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator1.accumulate()
    coco_evaluator1.summarize()


    print("Averaged stats2:", metric_logger)
    coco_evaluator2.synchronize_between_processes()

    # accumulate predictions from all images
    coco_evaluator2.accumulate()
    coco_evaluator2.summarize()


    torch.set_num_threads(n_threads)
    return coco_evaluator1, coco_evaluator2

@torch.no_grad()
def evaluate_compare_model_output(model1, model2, data_loader, device):
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model1.eval()
    model2.eval()

    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)

    iou_types1 = _get_iou_types(model1)
    coco_evaluator1 = CocoEvaluator(coco, iou_types1)

    iou_types2 = _get_iou_types(model2)
    coco_evaluator2 = CocoEvaluator(coco, iou_types2)

    idx = 0
    for image, targets in metric_logger.log_every(data_loader, 100, header):
        image = list(img.to(device) for img in image)
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        torch.cuda.synchronize()
        model_time = time.time()
        # print('-- image # {}'.format(idx))
        outputs1 = model1(image)
        outputs2 = model2(image)

        outputs1 = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs1]
        outputs2 = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs2]
        print('-- split output --')
        print(outputs1)

        print('-- orig output --')
        print(outputs2)
    return

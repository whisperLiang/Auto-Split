import distiller
import distiller.quantization as quantization
from obj_detection.models import resnet
import logging
import torchvision.models.detection as detection
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
# from obj_detection.pq_detection.ptq_compress import quantize_detection_model
import torch.nn as nn

import copy
# Logger handle
msglogger = logging.getLogger()


def compare_params_with_subset_dnn(subset_dnn, resnet50, start_idx=0):
    subset_dnn_params = dict(subset_dnn.named_parameters())
    resnet50_params = dict(resnet50.named_parameters())

    subset_dnn_stats = [(x, subset_dnn_params[x].data.shape) for x in subset_dnn_params]
    resnet50_stats = [(x, resnet50_params[x].data.shape) for x in resnet50_params]

    print('-- cloud stats --')
    subset_dnn_len = len(subset_dnn_stats)
    for idx, x1 in enumerate(subset_dnn_stats):
        print('{},{},{}'.format(idx, x1[0], x1[1]))

    print('-- orig stats --')

    for idx, x1 in enumerate(resnet50_stats):
        if idx < start_idx:
            continue

        if idx >= subset_dnn_len + start_idx:
            break
        print('{},{},{}'.format(idx, x1[0], x1[1]))

    return subset_dnn_len

def copy_params_from_parent_dnn(sub_dnn, resnet50, prefix=None, start_idx=0):
    no_match=0
    sub_dnn_params = dict(sub_dnn.named_parameters())
    sub_dnn_len = len(sub_dnn_params)
    resnet50_params = dict(resnet50.named_parameters())

    idx=0
    for name, W in resnet50.named_parameters():
        if idx < start_idx:
            continue

        if idx >= sub_dnn_len + start_idx:
            break
        if prefix is not None:
            wo_prefix_name = name.split(prefix)[-1]
        else:
            wo_prefix_name = name

        if wo_prefix_name in sub_dnn_params:
            edge_shape = list(sub_dnn_params[wo_prefix_name].data.shape)
            resnet50_shape = list(resnet50_params[name].data.shape)

            if edge_shape != resnet50_shape:
                print(wo_prefix_name)
                sub_dnn_params[wo_prefix_name].data.copy_(resnet50_params[name].data)

        else:
            no_match +=1

        idx+=1

    print('-- no match: {}'.format(no_match))
    return sub_dnn, no_match



if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(
        description=__doc__)
    parser.add_argument('--data-path', default='~/datasets/coco2017', help='dataset')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--model', default='fasterrcnn_resnet50_fpn', help='model')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
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

    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')
    args = parser.parse_args()

    criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
    renet_classification = resnet.resnet50(pretrained=False)
    resnet_edge = resnet.resnet_edge()
    edge_dnn_len = compare_params_with_subset_dnn(resnet_edge, renet_classification)
    resnet_edge, no_match_edge = copy_params_from_parent_dnn(resnet_edge, renet_classification)
    if no_match_edge !=0 : raise ValueError('parameters of edge dnn does not match orig dnn')

    print('-- cloud compare stats --')
    resnet_cloud = resnet.resnet_cloud()
    compare_params_with_subset_dnn(resnet_cloud, renet_classification, start_idx=edge_dnn_len)
    resnet_cloud, no_match_cloud = copy_params_from_parent_dnn(resnet_cloud, renet_classification,start_idx=edge_dnn_len)
    if no_match_cloud != 0: raise ValueError('parameters of cloud dnn does not match orig dnn')

import argparse
import json
import torch
from copy import deepcopy
from torch.utils.data import DataLoader
import logging
from functools import partial
from collections import OrderedDict

from detection_models.yolov3_master.distiller_models import Darknet as Darknet_distiller
from detection_models.yolov3_master.models import Darknet
from detection_models.yolov3_master.utils.datasets import *
from detection_models.yolov3_master.utils.utils import *
from detection_models.yolov3_master.utils.parse_config import parse_data_cfg

import distiller
from distiller.data_loggers import collect_quant_stats
import distiller.apputils as apputils


import distiller.quantization.ptq_coordinate_search as lapq

from distiller.quantization.range_linear_bit_search import PostTrainLinearQuantizerBitSearch, \
    RangeLinearEmbeddingWrapperBS, RangeLinearQuantParamLayerWrapperBS

from distiller.quantization.range_linear import ClipMode, \
    is_linear_quant_mode_asymmetric, is_linear_quant_mode_symmetric, is_post_train_quant_wrapper, LinearQuantMode

from distiller.model_transforms import fold_batch_norms
from distiller.quantization.ptq_bit_search import validate_quantization_settings, init_linear_quant_params

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

def evaluate(model, dataloader, is_training, coco91class, augment,conf_thres,
             iou_thres, multi_label, niou,iouv, names, save_json, device):
    seen=0
    s = ('%20s' + '%10s' * 6) % ('Class', 'Images', 'Targets', 'P', 'R', 'mAP@0.5', 'F1')
    t0, t1 = 0., 0.
    loss = torch.zeros(3, device=device)
    jdict, stats, ap, ap_class = [], [], [], []
    for batch_i, (imgs, targets, paths, shapes) in enumerate(tqdm(dataloader, desc=s)):
        imgs = imgs.to(device).float() / 255.0  # uint8 to float32, 0 - 255 to 0.0 - 1.0
        targets = targets.to(device)
        nb, _, height, width = imgs.shape  # batch size, channels, height, width
        whwh = torch.Tensor([width, height, width, height]).to(device)

        # Disable gradients
        with torch.no_grad():
            # Run model
            t = torch_utils.time_synchronized()
            inf_out, train_out = model(imgs, augment=augment)  # inference and training outputs
            t0 += torch_utils.time_synchronized() - t

            # Compute loss
            if is_training:  # if model has loss hyperparameters
                loss += compute_loss(train_out, targets, model)[1][:3]  # GIoU, obj, cls

            # Run NMS
            t = torch_utils.time_synchronized()
            output = non_max_suppression(inf_out, conf_thres=conf_thres, iou_thres=iou_thres, multi_label=multi_label)
            t1 += torch_utils.time_synchronized() - t

        # Statistics per image
        for si, pred in enumerate(output):
            labels = targets[targets[:, 0] == si, 1:]
            nl = len(labels)
            tcls = labels[:, 0].tolist() if nl else []  # target class
            seen += 1

            if pred is None:
                if nl:
                    stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                continue

            # Append to text file
            # with open('test.txt', 'a') as file:
            #    [file.write('%11.5g' * 7 % tuple(x) + '\n') for x in pred]

            # Clip boxes to image bounds
            clip_coords(pred, (height, width))

            # Append to pycocotools JSON dictionary
            if save_json:
                # [{"image_id": 42, "category_id": 18, "bbox": [258.15, 41.29, 348.26, 243.78], "score": 0.236}, ...
                image_id = int(Path(paths[si]).stem.split('_')[-1])
                box = pred[:, :4].clone()  # xyxy
                scale_coords(imgs[si].shape[1:], box, shapes[si][0], shapes[si][1])  # to original shape
                box = xyxy2xywh(box)  # xywh
                box[:, :2] -= box[:, 2:] / 2  # xy center to top-left corner
                for p, b in zip(pred.tolist(), box.tolist()):
                    jdict.append({'image_id': image_id,
                                  'category_id': coco91class[int(p[5])],
                                  'bbox': [round(x, 3) for x in b],
                                  'score': round(p[4], 5)})

            # Assign all predictions as incorrect
            correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool, device=device)
            if nl:
                detected = []  # target indices
                tcls_tensor = labels[:, 0]

                # target boxes
                tbox = xywh2xyxy(labels[:, 1:5]) * whwh

                # Per target class
                for cls in torch.unique(tcls_tensor):
                    ti = (cls == tcls_tensor).nonzero().view(-1)  # prediction indices
                    pi = (cls == pred[:, 5]).nonzero().view(-1)  # target indices

                    # Search for detections
                    if pi.shape[0]:
                        # Prediction to target ious
                        ious, i = box_iou(pred[pi, :4], tbox[ti]).max(1)  # best ious, indices

                        # Append detections
                        for j in (ious > iouv[0]).nonzero():
                            d = ti[i[j]]  # detected target
                            if d not in detected:
                                detected.append(d)
                                correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                if len(detected) == nl:  # all targets already located in image
                                    break

            # Append statistics (correct, conf, pcls, tcls)
            stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        # Plot images
        if batch_i < 1:
            f = 'test_batch%g_gt.jpg' % batch_i  # filename
            plot_images(imgs, targets, paths=paths, names=names, fname=f)  # ground truth
            f = 'test_batch%g_pred.jpg' % batch_i
            plot_images(imgs, output_to_target(output, width, height), paths=paths, names=names, fname=f)  # predictions

    return stats, t0,t1, jdict, loss, seen


def test(cfg,
         data, cocodir_path, args,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=False,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True):
    msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir,
                                         args.verbose)
    distiller_model = None
    # Initialize/load model and set device
    if model is None:
        is_training = False
        device = torch_utils.select_device(opt.device_yolo, batch_size=batch_size)
        verbose = opt.task == 'test'

        # Remove previous
        for f in glob.glob('test_batch*.jpg'):
            os.remove(f)

        # Initialize model
        model = Darknet(cfg, imgsz)
        # distiller_model = Darknet_distiller(cfg,imgsz)
        if weights.endswith('.pt'):  # pytorch format
            model.load_state_dict(torch.load(weights, map_location=device)['model'])
        #     len_edge_dnn = len(distiller_model.edge_dnn.module_list)
        #     idx=0
        #     orig_dict = model.state_dict()
        #
        #     #---- Copy EDGE DNN Params ---
        #     edge_dict = distiller_model.edge_dnn.state_dict()
        #     # 1. filter out unnecessary keys
        #     orig_edge_dict = {k: v for k, v in orig_dict.items() if k in edge_dict}
        #     # 2. overwrite entries in the existing state dict
        #     edge_dict.update(orig_edge_dict)
        #     # 3. load the new state dict
        #     distiller_model.edge_dnn.load_state_dict(orig_edge_dict)
        #
        #     # ---- Copy Cloud DNN Params ---
        #     cloud_dict = distiller_model.cloud_dnn.state_dict()
        #     # 1. filter out unnecessary keys
        #     orig_cloud_dict = {k: v for k, v in orig_dict.items() if k not in edge_dict}
        #
        #     def get_new_cloud_key(old_name,len_edge_dnn):
        #         old_name_list = old_name.split('.')
        #         y = ''
        #         isfirstno = True
        #         for i_l, x in enumerate(old_name_list):
        #             if isfirstno and x.isnumeric():
        #                 isfirstno = False
        #                 if i_l == 0:
        #                     y += str(int(x) - len_edge_dnn)
        #                 else:
        #                     y = y + '.' + str(int(x) - len_edge_dnn)
        #
        #             else:
        #                 if i_l == 0:
        #                     y += x
        #                 else:
        #                     y = y + '.' + x
        #         return y
        #
        #     # Map old cloud keys to new keys:
        #     new_cloud_dict = OrderedDict()
        #     for old_key, old_item in orig_cloud_dict.items():
        #         new_key = get_new_cloud_key(old_key, len_edge_dnn)
        #         new_cloud_dict[new_key] = old_item
        #
        #     # 2. Verify if all keys match in state dict
        #     assert len(new_cloud_dict) == len(cloud_dict),"Cloud Keys do not match "
        #     for k,v in cloud_dict.items():
        #         if k not in new_cloud_dict:
        #             return ValueError('Cloud state_dict Keys do not match ')
        #     # 3. load the new state dict
        #     distiller_model.cloud_dnn.load_state_dict(new_cloud_dict)
        #
        #
        # else:
        #     raise Exception('Only pytorch .pt model supported')
        # # Load weights
        # attempt_download(weights)
        # if weights.endswith('.pt'):  # pytorch format
        #     model.load_state_dict(torch.load(weights, map_location=device)['model'])
        # else:  # darknet format
        #     load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        model.to(device)

        # distiller_model.fuse()
        # distiller_model.to(device)

    else:  # called by train.py
        is_training = True
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images
    names = load_classes(data['names'])  # class names
    iouv = torch.linspace(0.5, 0.95, 10).to(device)  # iou vector for mAP@0.5:0.95
    iouv = iouv[0].view(1)  # comment for mAP@0.5:0.95
    niou = iouv.numel()

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, imgsz, batch_size, rect=True, single_cls=opt.single_cls, pad=0.5)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                # num_workers=4,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    model.eval()
    x1,p1 = model(torch.zeros((1, 3, imgsz, imgsz), device=device)) if device.type != 'cpu' else None  # run once
    coco91class = coco80_to_coco91_class()

    stats, t0, t1, jdict, loss, seen = evaluate(model, dataloader, is_training, coco91class,
                     augment, conf_thres, iou_thres, multi_label,
                     niou, iouv, names, save_json, device)

    # Compute statistics
    p, r, f1, mp, mr, map, mf1 = 0., 0., 0., 0., 0., 0., 0.
    stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
    if len(stats):
        p, r, ap, f1, ap_class = ap_per_class(*stats)
        if niou > 1:
            p, r, ap, f1 = p[:, 0], r[:, 0], ap.mean(1), ap[:, 0]  # [P, R, AP@0.5:0.95, AP@0.5]
        mp, mr, map, mf1 = p.mean(), r.mean(), ap.mean(), f1.mean()
        nt = np.bincount(stats[3].astype(np.int64), minlength=nc)  # number of targets per class
    else:
        nt = torch.zeros(1)

    # Print results
    pf = '%20s' + '%10.3g' * 6  # print format
    print(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            print(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        print('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        print('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        with open('results.json', 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval

            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            coco_dir = os.path.abspath(cocodir_path)
            annotation_path = os.path.join(coco_dir,'annotations/instances_val*.json')
            cocoGt = COCO(glob.glob(annotation_path)[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes('results.json')  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except:
            print('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]
    return (mp, mr, map, mf1, *(loss.cpu() / len(dataloader)).tolist()), maps




def init_yolo_detection_arg_parser(parser=None,include_ptq_lapq_args=False):
    '''Object detection compression application command-line arguments.
    '''

    QUANT_CHOICES = ['bit_search', 'fake_pq', 'pq', 'eval']

    if parser is None:
        parser = argparse.ArgumentParser(description='Distiller image classification model compression')

    # parser.add_argument('--data-path', default='~/datasets/coco2017', help='dataset')
    # parser.add_argument('data', metavar='DATASET_DIR', help='path to dataset', default='~/datasets/coco2017')
    parser.add_argument('--dataset', default='coco', help='dataset')
    parser.add_argument('--quant-method', type=lambda s: s.lower(), choices=QUANT_CHOICES, default='bit_search',
                        help='print a summary of the model, and exit - options: | '.join(QUANT_CHOICES))

    # parser.add_argument('--model', default='yolov3', help='model')
    # parser.add_argument('--arch', '-a', metavar='ARCH', default='yolov3', type=lambda s: s.lower())
                        # choices=models.ALL_MODEL_NAMES,
                        # help='model architecture: ' +
                        # ' | '.join(models.ALL_MODEL_NAMES) +
                        # ' (default: fasterrcnn_resnet50_fpn)')

    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=1, type=int)
    parser.add_argument('--epochs', default=128, type=int, metavar='N',
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

if __name__ == '__main__':
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    parser = argparse.ArgumentParser(prog='quantize_edge_dnn.py')
    parser.add_argument('--cfg', type=str, default='detection_models/yolov3_master/cfg/yolov3-tiny.cfg', help='*.cfg path')
    parser.add_argument('--data', type=str, default='detection_models/yolov3_master/data/coco2017.data', help='*.data path')
    parser.add_argument('--weights', type=str, default='<path to >/yolov3_data/yolov3-tiny.pt', help='weights path')
    parser.add_argument('--cocodir', type=str, default='/data/datasets/coco2017/',
                        help='coco2017 dir path')
    # parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')
    parser.add_argument('--img-size', type=int, default=512, help='inference size (pixels): From source 320,416,512,608')
    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device_yolo', default='1', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser =  init_yolo_detection_arg_parser(parser=parser, include_ptq_lapq_args=True)
    args = parser.parse_args()

    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    print(opt)


    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        # ------------------------------------------------------------------------------------------------
        device = torch.device(args.device)
        script_dir = os.path.dirname(__file__)
        # if not os.path.exists(args.output_dir):
        #     os.makedirs(args.output_dir)
        # if distiller_utils.is_main_process():
        #     msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir,
        #                                          args.verbose)
        #
        #     # Log various details about the execution environment.  It is sometimes useful
        #     # to refer to past experiment executions and this information may be useful.
        #     apputils.log_execution_env_state(
        #         filter(None, [args.compress, args.qe_stats_file]),  # remove both None and empty strings
        #         msglogger.logdir)
        #     msglogger.debug("Distiller: %s", distiller.__version__)
        # else:
        #     msglogger = logging.getLogger()
        #     msglogger.disabled = True

        criterion = nn.CrossEntropyLoss(reduction='none').to(args.device)
        # ------------------------------------------------------------------------------------------------

        test(opt.cfg,
             opt.data, opt.cocodir, args,
             opt.weights,
             args.batch_size,
             opt.img_size,
             opt.conf_thres,
             opt.iou_thres,
             opt.save_json,
             opt.single_cls,
             opt.augment)

    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

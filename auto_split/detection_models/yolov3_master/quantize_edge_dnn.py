import argparse
import json
import sys
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
from detection_models.yolov3_master.test_orig import evaluate


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


msglogger = logging.getLogger()
print = msglogger.info

def yolo_ptq_bit_search(quantizer, dummy_input, eval_fn, method='Powell',
                          maxiter=None, maxfev=None, basinhopping=False, basinhopping_niter=100,
                          init_mode=ClipMode.NONE, init_method=None, search_clipping=False,
                          minimizer_kwargs=None, is_eval=True):
    """
    Searches for the optimal post-train quantization configuration (scale/zero_points)
    for a model using numerical methods, as described by scipy.optimize.minimize.
    Args:
        quantizer (distiller.quantization.PostTrainLinearQuantizer): A configured PostTrainLinearQuantizer object
          containing the model being quantized
        dummy_input: an sample expected input to the model
        eval_fn (callable): evaluation function for the model. Assumed it has a signature of the form
          `eval_fn(model)->float`. this is the function to be minimized by the optimization algorithm.
        test_fn (callable): a function to test the current performance of the model. Assumed it has a signature of
          the form `test_fn(model)->dict`, where the returned dict contains relevant results to be logged.
          For example: {'top-1': VAL, 'top-5': VAL, 'loss': VAL}
        method (str or callable): Minimization method as accepted by scipy.optimize.minimize.
        maxiter (int): Maximum number of iterations to perform during minimization
        maxfev (int): Maximum number of total function evaluations to perform during minimization
        basinhopping (bool): flag, indicates to use basinhopping as a global-minimization method,
          will pass the `method` argument to `scipy.optimize.basinhopping`.
        basinhopping_niter (int): Number of iterations to perform if basinhopping is set
        init_mode (ClipMode or callable or str or dict): See 'init_linear_quant_params'
        init_method (str or callable): See 'init_layer_linear_quant_params'
        search_clipping (bool): Search on clipping values instead of directly on scale/zero-point (scale and zero-
          point are inferred from the clipping values)
        minimizer_kwargs (dict): Optional additional arguments for scipy.optimize.minimize
    """
    msglogger = logging.getLogger()
    if not isinstance(quantizer, PostTrainLinearQuantizerBitSearch):
        raise ValueError('Only PostTrainLinearQuantizerBitSearch supported, but got a {}'.format(quantizer.__class__.__name__))
    if quantizer.prepared:
        raise ValueError('Expecting a quantizer for which prepare_model has not been called')

    original_model = deepcopy(quantizer.model)
    original_model = fold_batch_norms(original_model, dummy_input)

    if not quantizer.model_activation_stats:
        msglogger.info('Collecting stats for model...')
        model_temp = distiller.utils.make_non_parallel_copy(original_model)
        act_stats = collect_quant_stats(model_temp, eval_fn,
                                        inplace_runtime_check=True, disable_inplace_attrs=True,
                                        save_dir=getattr(msglogger, 'logdir', '.'))
        del model_temp
        quantizer.model_activation_stats = act_stats
        quantizer.model.quantizer_metadata['params']['model_activation_stats'] = act_stats

    # Preparing model and init conditions:
    msglogger.info("Initializing quantizer...")

    # Make sure weights are re-quantizable and clip-able
    quantizer.save_fp_weights = True
    quantizer.also_clip_weights = True

    # Disable any user set activations clipping - we'll be using init_args
    quantizer.clip_acts = ClipMode.NONE
    for overrides_dict in quantizer.module_overrides_map.values():
        overrides_dict.pop('clip_acts', None)

    quantizer.prepare_model(dummy_input)
    quantizer.model.eval()

    validate_quantization_settings(quantizer.model, search_clipping)

    msglogger.info("Initializing quantization parameters...")
    init_linear_quant_params(quantizer, original_model, eval_fn, dummy_input, init_mode, init_method,
                             search_clipping=search_clipping)
    # Note: when collecting activation stats is_eval=False
    # else the data will be overwritten for each image
    # This can cause problems as input shapes of images are different in coco
    # and dummy input assumes an X*X*3 shape. For example, 416x416x3,
    # thus, out_act/ data will not match  post_prepared_df.csv
    if not quantizer.model_activation_stats and is_eval:
        features = eval_fn(quantizer.model)
    else:
        features = None
    # msglogger(len(features), features[0].shape)
    return quantizer.model, features

def edge_evaluate(model, dataloader, augment):
    def augment_images(x):
        img_size = x.shape[-2:]  # height, width
        s = [0.83, 0.67]  # scales
        y = []
        for i, xi in enumerate((x,
                                torch_utils.scale_img(x.flip(3), s[0], same_shape=False),  # flip-lr and scale
                                torch_utils.scale_img(x, s[1], same_shape=False),  # scale
                                )):
            y.append(xi)

        return y

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
            if augment:
                list_imgs = augment_images(imgs)
                for im in list_imgs:
                    img_out = model(imgs)  # inference and training outputs
            else:
                img_out = model(imgs)  # inference and training outputs
    return

def copy_params_to_dnns(model, weights, distiller_model):
    # 0. Load Original model state dict
    model.load_state_dict(torch.load(weights, map_location=device)['model'])
    orig_dict = model.state_dict()

    # 1. find original edge and cloud dict items
    old_len_edge_dnn = distiller_model.num_old_edge_dnn_layers

    def filter_orig_edge_keys(orig_dict):
        orig_edge_dict = OrderedDict()
        orig_cloud_dict = OrderedDict()
        for old_name, v in orig_dict.items():
            old_name_list = old_name.split('.')
            for x in old_name_list:
                if x.isnumeric():
                    if int(x) < old_len_edge_dnn:
                        orig_edge_dict[old_name] = v
                    else:
                        orig_cloud_dict[old_name] = v

                    break
        return orig_edge_dict, orig_cloud_dict

    orig_edge_dict, orig_cloud_dict = filter_orig_edge_keys(orig_dict)

    # 2.1 Edge DNN: Map old Edge keys to new keys
    def get_new_keys(old_name):
        old_name_list = old_name.split('.')
        y = ''
        isfirstno = True
        for i_l, x in enumerate(old_name_list):
            if isfirstno and x.isnumeric():
                isfirstno = False

                new_x_list = distiller_model.old_name_mapping[int(x)]
                edge_dnn_idx, new_x_idx = new_x_list[0].split('.')

                if len(new_x_list) == 1:
                    new_x = new_x_idx
                else:
                    new_x = str(new_x_idx) + '.' + str(new_x_list[1])
                if i_l == 0:
                    y += str(new_x)
                else:
                    y = y + '.' + new_x
            else:
                if i_l == 0:
                    y += x
                else:
                    y = y + '.' + x
        return y

    new_edge_dict = OrderedDict()
    for old_key, old_item in orig_edge_dict.items():
        new_key = get_new_keys(old_key)
        new_edge_dict[new_key] = old_item

    # 2.2 Edge DNN: Check all keys are found
    edge_dict = distiller_model.edge_dnn.state_dict()
    if not set(edge_dict.keys()) == set(new_edge_dict.keys()):
        assert 0, 'Edge DNN Keys do not match'

    # 2.3 Edge DNN: Load state dict
    distiller_model.edge_dnn.load_state_dict(new_edge_dict)

    # 3.1 Cloud DNN: Map old Edge keys to new keys
    new_cloud_dict = OrderedDict()
    for old_key, old_item in orig_cloud_dict.items():
        new_key = get_new_keys(old_key)
        new_cloud_dict[new_key] = old_item

    # 3.2 Cloud DNN: Check all keys are found
    cloud_dict = distiller_model.cloud_dnn.state_dict()
    if not set(new_cloud_dict.keys()) == set(cloud_dict.keys()):
        assert 0, 'Cloud DNN Keys do not match'

    # 3.3 Cloud DNN: Load state dict
    distiller_model.cloud_dnn.load_state_dict(new_cloud_dict)

    return distiller_model

def test(cfg, script_dir,
         data, cocodir_path, args,
         weights=None,
         batch_size=16,
         imgsz=416,
         conf_thres=0.001,
         iou_thres=0.6,  # for nms
         save_json=True,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None,
         multi_label=True):

    distiller_model = model
    is_training = False
    device = torch_utils.select_device(opt.device_yolo, batch_size=batch_size)
    verbose = opt.task == 'test'

    # Remove previous
    for f in glob.glob('test_batch*.jpg'):
        os.remove(f)

    distiller_model = distiller_model.eval()
    distiller_model.to(device)


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
        dataset = LoadImagesAndLabels(path, cocodir_path, imgsz, batch_size, rect=True, single_cls=opt.single_cls, pad=0.5)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                # num_workers=4,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)


    distiller_model.eval()
    coco91class = coco80_to_coco91_class()

    stats, t0, t1, jdict, loss, seen = evaluate(distiller_model, dataloader, is_training, coco91class,
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
    msglogger.info(pf % ('all', seen, nt.sum(), mp, mr, map, mf1))

    # Print results per class
    if verbose and nc > 1 and len(stats):
        for i, c in enumerate(ap_class):
            msglogger.info(pf % (names[c], seen, nt[c], p[i], r[i], ap[i], f1[i]))

    # Print speeds
    if verbose or save_json:
        t = tuple(x / seen * 1E3 for x in (t0, t1, t0 + t1)) + (imgsz, imgsz, batch_size)  # tuple
        msglogger.info('Speed: %.1f/%.1f/%.1f ms inference/NMS/total per %gx%g image at batch-size %g' % t)

    # Save JSON
    if save_json and map and len(jdict):
        msglogger.info('\nCOCO mAP with pycocotools...')
        imgIds = [int(Path(x).stem.split('_')[-1]) for x in dataloader.dataset.img_files]
        results_json = os.path.join(os.getcwd(), msglogger.logdir, 'results.json')
        with open(results_json, 'w') as file:
            json.dump(jdict, file)

        try:
            from pycocotools.coco import COCO
            from pycocotools.cocoeval import COCOeval


            # https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
            coco_dir = os.path.abspath(cocodir_path)
            annotation_path = os.path.join(coco_dir,'annotations/instances_val*.json')
            cocoGt = COCO(glob.glob(annotation_path)[0])  # initialize COCO ground truth api
            cocoDt = cocoGt.loadRes(results_json)  # initialize COCO pred api

            cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
            cocoEval.params.imgIds = imgIds  # [:32]  # only evaluate these images
            cocoEval.evaluate()
            cocoEval.accumulate()
            results_file = os.path.join(os.getcwd(), msglogger.logdir, 'map.txt')
            print = msglogger.info
            log = open(results_file, "a")
            sys.stdout = log
            cocoEval.summarize()
            log.close()
            # mf1, map = cocoEval.stats[:2]  # update to pycocotools results (mAP@0.5:0.95, mAP@0.5)
        except:
            msglogger.info('WARNING: pycocotools must be installed with numpy==1.17 to run correctly. '
                  'See https://github.com/cocodataset/cocoapi/issues/356')

    # Return results
    maps = np.zeros(nc) + map
    for i, c in enumerate(ap_class):
        maps[c] = ap[i]

    yy = loss.cpu() / len(dataloader)
    zz = yy.tolist()
    xx = mp, mr, map, mf1, *(zz)
    msglogger.info(xx)
    return (xx), maps


def ptq_edge_dnn(cfg,
         data, cocodir_path, args,
         weights=None,
         batch_size=16,
         imgsz=416,
         single_cls=False,
         augment=False,
         model=None,
         dataloader=None):
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
        distiller_model = Darknet_distiller(cfg,msglogger, imgsz)
        distiller_model = distiller_model.eval()
        if weights.endswith('.pt'):  # pytorch format
            distiller_model = copy_params_to_dnns(model, weights, distiller_model)
        else:
            raise Exception('Only pytorch .pt model supported')
        # # Load weights
        # attempt_download(weights)
        # if weights.endswith('.pt'):  # pytorch format
        #     model.load_state_dict(torch.load(weights, map_location=device)['model'])
        # else:  # darknet format
        #     load_darknet_weights(model, weights)

        # Fuse
        model.fuse()
        model.to(device)

        distiller_model.fuse()
        distiller_model.to(device)

    else:  # called by train.py
        is_training = True
        device = next(model.parameters()).device  # get model device
        verbose = False

    # Configure run
    data = parse_data_cfg(data)
    nc = 1 if single_cls else int(data['classes'])  # number of classes
    path = data['valid']  # path to test images

    # Dataloader
    if dataloader is None:
        dataset = LoadImagesAndLabels(path, cocodir_path, imgsz, batch_size, rect=True, single_cls=opt.single_cls, pad=0.5)
        batch_size = min(batch_size, len(dataset))
        dataloader = DataLoader(dataset,
                                batch_size=batch_size,
                                num_workers=min([os.cpu_count(), batch_size if batch_size > 1 else 0, 8]),
                                # num_workers=4,
                                pin_memory=True,
                                collate_fn=dataset.collate_fn)

    model.eval()
    input = torch.rand((1, 3, imgsz, imgsz), device=device)
    x1,p1 = model(input) if device.type != 'cpu' else None  # run once

    distiller_model.eval()
    x2,p2 = distiller_model(input) if device.type != 'cpu' else None  # run once
    diff_x =  (x2 - x1).sum()
    diff_p = []
    for i, val1 in enumerate(p1):
        x = p1[i] - p2[i]
        diff_p.append(x.sum())

    dummy_input = torch.rand(1, 3, imgsz, imgsz)
    dummy_input = dummy_input.to(device)
    test_fn = partial(edge_evaluate, dataloader=dataloader, augment=augment)
  #-----------------------------------------------------------------------------------------
    if args.quant_method == 'bit_search':
        msglogger.info('Bit Search: evaluating Quantized Edge DNN with stat collection  ')
        quantizer = distiller.quantization.PostTrainLinearQuantizerBitSearch.from_args(distiller_model.edge_dnn, args)
        edge_quantized_model,_ = yolo_ptq_bit_search(quantizer, dummy_input, test_fn,
                       **lapq.cmdline_args_to_dict(args), is_eval=False)
        distiller_model.edge_dnn = edge_quantized_model
        return distiller_model
    elif args.quant_method == 'fake_pq':
        msglogger.info('PTQ Fake Quantization: evaluating Quantized Edge DNN')
        quantizer = distiller.quantization.PostTrainLinearQuantizer.from_args(distiller_model.edge_dnn, args)
        # edge_quantized_model,_ = yolo_prepare_edge_dnn(quantizer, dummy_input, test_fn,
        #                **lapq.cmdline_args_to_dict(args), is_eval=True)
        # distiller_model.edge_dnn = edge_quantized_model
        edge_quantized_model,_ = lapq.ptq_fake_quantization(quantizer, dummy_input, test_fn, test_fn=test_fn,
                                                     **lapq.cmdline_args_to_dict(args), is_eval=False)
        distiller_model.edge_dnn = edge_quantized_model
        return distiller_model
    # elif args.quant_method == 'pq':
    #     msglogger.info('PTQ: evaluating Quantized Edge DNN')
    #     quantizer, features = self.quantize_detection_model(model, criterion, test_fn, args,
    #                                                         test_data_loader, loggers=None, save_flag=True,
    #                                                         is_eval=is_eval)
    #     return quantizer.model, features
    elif args.quant_method == 'eval':
        msglogger.info('evaluating Edge Float model (not quantized)')
        model.to(args.device)
        # if is_eval:
        #     features = test_fn(model)
        # else:
        #     features = None
        return distiller_model
    else:
        raise ValueError('wrong args.quant_method .. Given {}'.format(args.quant_method))

  #-----------------------------------------------------------------------------------------



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
    parser.add_argument('-b', '--batch-size', default=16, type=int)
    parser.add_argument('--epochs', default=1, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('--start-epoch', default=0, type=int, help='starting epoch number')
    parser.add_argument('-j', '--workers', default=8, type=int, metavar='N',
                        help='number of data loading workers (default: 4), 0 for debug')

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
    # Yolo 416
    parser = argparse.ArgumentParser(prog='quantize_edge_dnn.py')
    parser.add_argument('--cfg', type=str, default='detection_models/yolov3_master/cfg/yolov3.cfg', help='*.cfg path')
    parser.add_argument('--weights', type=str, default='<path to >/yolov3_data/yolov3.pt', help='weights path')
    parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels): From source 320,416,512,'
                                                                  '608')

    # # Tiny Yolo
    # parser = argparse.ArgumentParser(prog='quantize_edge_dnn.py')
    # parser.add_argument('--cfg', type=str, default='detection_models/yolov3_master/cfg/yolov3-tiny.cfg', help='*.cfg path')
    # parser.add_argument('--data', type=str, default='detection_models/yolov3_master/data/coco2017.data', help='*.data path')
    # parser.add_argument('--weights', type=str, default='<path to >/yolov3_data/yolov3-tiny.pt', help='weights path')
    # parser.add_argument('--img-size', type=int, default=416, help='inference size (pixels): From source 320,416,512,
    # 608')

    parser.add_argument('--data', type=str, default='detection_models/yolov3_master/data/coco2017.data', help='*.data path')
    parser.add_argument('--cocodir', type=str, default='/data/datasets/coco2017/',
                        help='coco2017 dir path')
    # parser.add_argument('--batch-size', type=int, default=16, help='size of each image batch')

    parser.add_argument('--conf-thres', type=float, default=0.001, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.6, help='IOU threshold for NMS')
    parser.add_argument('--save-json', action='store_true', help='save a cocoapi-compatible JSON results file')
    parser.add_argument('--task', default='test', help="'test', 'study', 'benchmark'")
    parser.add_argument('--device_yolo', help='device id (i.e. 0 or 0,1) or cpu')
    parser.add_argument('--single-cls', action='store_true', help='train as single-class dataset')
    parser.add_argument('--augment', action='store_true', help='augmented inference')

    parser =  init_yolo_detection_arg_parser(parser=parser, include_ptq_lapq_args=True)
    args = parser.parse_args()

    opt = parser.parse_args()
    opt.save_json = opt.save_json or any([x in opt.data for x in ['coco.data', 'coco2014.data', 'coco2017.data']])
    opt.cfg = check_file(opt.cfg)  # check file
    opt.data = check_file(opt.data)  # check file
    # msglogger = logging.getLogger('Quantize-Edge-DNN: ')



    # task = 'test', 'study', 'benchmark'
    if opt.task == 'test':  # (default) test normally
        # ------------------------------------------------------------------------------------------------
        device = torch.device(args.device)
        script_dir = os.path.dirname(__file__)
        msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir,
                                             args.verbose)
        msglogger.info(opt)

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
        distiller_model = ptq_edge_dnn(opt.cfg,
                                 opt.data, opt.cocodir, args,
                                 opt.weights,
                                 args.batch_size,
                                 opt.img_size,
                                 opt.single_cls,
                                 opt.augment)

        if args.quant_method != 'bit_search':
            test(opt.cfg, script_dir,
                 opt.data, opt.cocodir,
                 args,
                 opt.weights,
                 args.batch_size,
                 opt.img_size,
                 opt.conf_thres,
                 opt.iou_thres,
                 opt.save_json,
                 opt.single_cls,
                 opt.augment, model=distiller_model)


    elif opt.task == 'benchmark':  # mAPs at 256-640 at conf 0.5 and 0.7
        script_dir = os.path.dirname(__file__)
        msglogger = apputils.config_pylogger(os.path.join(script_dir, 'logging.conf'), args.name, args.output_dir,
                                             args.verbose)
        msglogger.info(opt)
        y = []
        for i in list(range(256, 640, 128)):  # img-size
            for j in [0.6, 0.7]:  # iou-thres
                t = time.time()
                r = test(opt.cfg, opt.data, opt.weights, opt.batch_size, i, opt.conf_thres, j, opt.save_json)[0]
                y.append(r + (time.time() - t,))
        np.savetxt('benchmark.txt', y, fmt='%10.4g')  # y = np.loadtxt('study.txt')

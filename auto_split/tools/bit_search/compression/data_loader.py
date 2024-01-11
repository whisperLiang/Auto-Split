from attrdict import AttrDict
import random
import numpy as np
import os, copy
import datetime
from pathlib import Path
import pandas as pd
from tools.hw_simulator.schedule.net import Net
from tools.hw_simulator.schedule.dnn_schedules.hwcf.hwcf_schedule import HWCFSchedule
from tools.bit_search.memory_constraint_solver import MemorySolver
from collections import OrderedDict
import time
import itertools
from operator import itemgetter
import logging, sys
from tools.bit_search.model_latency_2_gen import ModelLatency
import yaml
from PIL import Image
from torchvision import transforms
# import torchvision.models as models
import timeit
import torch
import tools.hw_simulator.hw_utility.model_summary as model_summary
import os
from pathlib import Path
import time
from copy import deepcopy
import argparse
from  tools.bit_search.dnn_quantizer import DNNQuantizer
from tools.bit_search.post_train_quantizer import add_post_train_quant_args
from distiller import models
# from examples.classifier_compression import
# from distiller.utils import float_range_argparse_checker as float_range
#
from examples.classifier_compression import parser
import distiller.apputils.image_classifier as classifier
from examples.classifier_compression import *
from distiller.models import create_model

class DataLoader():

    def __init__(self,model_name, input_stats_file, bitlist, bitlist_cloud, model_activation_stats,args):
        # Set up logging
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.logger_auto = logging.getLogger('Bit-Solver')
        orig_model = create_model(pretrained=True, dataset='ImageNet', arch='resnet50', parallel=False, device_ids=-1)
        # model_cmd = 'models.' + model_name + '(pretrained=True)'
        # orig_model = eval(model_cmd)
        # orig_model.eval()


        # Set up input_stats_df
        self.input_stats_df = pd.read_csv(input_stats_file)
        NUMLAYERS = len(self.input_stats_df)
        # NUMLAYERS = 5
        self.input_stats_df = self.input_stats_df[0:NUMLAYERS]
        self.model_name = model_name
        # self.load_weights_stats(orig_model, NUMLAYERS)
        # self.load_activation_stats(model_activation_stats)
        # self.process_activation_stats()

        self.quantizer = DNNQuantizer(orig_model, args)

        return


    def process_activation_stats(self):
        idx=0
        act_stats_list = list(self.model_activation_stats.items())
        total_keys = len(act_stats_list)
        for key, value in self.model_activation_stats.items():
            if key in self.dnn_weights_d.layers_df.index:
                bn_idx = idx+1
                relu_idx = idx+2
                if bn_idx < total_keys and relu_idx < total_keys \
                        and 'bn' in act_stats_list[bn_idx][0] \
                        and 'relu' in act_stats_list[relu_idx][0]:

                    relu_value = act_stats_list[relu_idx][1]
                    relu_out_min = relu_value['output']['min']
                    relu_out_max = relu_value['output']['max']
                    print('{} rmin: {} rmax: {}'.format(key,relu_out_min,relu_out_max))
                else:
                    print('{} min: {} max: {}'.format(key, value['output']['min'], value['output']['max']))
                    self.logger_auto.warning('no bn,relu layer after layer: {}'.format(key))
            idx+=1

        return

    def load_activation_stats(self, model_activation_stats):
        # Set up activation stats
        if model_activation_stats is not None:
            if isinstance(model_activation_stats, str):
                if not os.path.isfile(model_activation_stats):
                    raise ValueError("Model activation stats file not found at: " + model_activation_stats)
                self.logger_auto.info('Loading activation stats from: ' + model_activation_stats)
                with open(model_activation_stats, 'r') as stream:
                    self.model_activation_stats = self.yaml_ordered_load(stream)
            elif not isinstance(model_activation_stats, (dict, OrderedDict)):
                raise TypeError('model_activation_stats must either be a string, a dict / OrderedDict or None')

        print('')
        return

    def load_weights_stats(self, model, num_layers):
        # Load model
        dnn_weights_d = AttrDict()
        dnn_weights_d['vol'] = OrderedDict()
        layer_names_df = self.input_stats_df['name']
        dnn_weights_d.layers_df = pd.DataFrame(index=layer_names_df,
                                               columns = ['vol', 'sat_min','sat_max','orig_layer_idx'])
        dnn_weights_d.min_max = AttrDict()

        # model.to('cuda')

        # Load Model Weights
        layer_idx = 0
        sel_layer_idx = 0
        for name, W in model.named_parameters():
            if sel_layer_idx == num_layers:
                break

            if 'weight' in name:
                layer_name = name.split('.weight')[0]
                if layer_name in dnn_weights_d.layers_df.index:
                    W.requires_grad=False
                    W = W.view(-1)
                    dnn_weights_d['vol'][layer_name] = W.numpy()
                    dnn_weights_d.layers_df.loc[layer_name] = { 'vol': W.shape[0],
                                                                'sat_min': W.min().numpy(),
                                                                'sat_max': W.max().numpy(),
                                                                'orig_layer_idx':layer_idx
                                                                }
                    sel_layer_idx += 1
                else:
                    self.logger_auto.warning('not found: {}'.format(layer_name))

                layer_idx+=1

        self.dnn_weights_d = dnn_weights_d
        return

    def yaml_ordered_load(self, stream, Loader=yaml.Loader, object_pairs_hook=OrderedDict):
        """Function to load YAML file using an OrderedDict

        See: https://stackoverflow.com/questions/5121931/in-python-how-can-you-load-yaml-mappings-as-ordereddicts
        """
        class OrderedLoader(Loader):
            pass

        def construct_mapping(loader, node):
            loader.flatten_mapping(node)
            return object_pairs_hook(loader.construct_pairs(node))

        OrderedLoader.add_constructor(
            yaml.resolver.BaseResolver.DEFAULT_MAPPING_TAG,
            construct_mapping)

        return yaml.load(stream, OrderedLoader)


def main():

if __name__ == '__main__':

    print(datetime.datetime.now())
    model_names = ['CONV2D', 'resnet18', 'resnet50', 'mobilenet_v2', 'densenet201']
    device = 'cloud'
    root_dir = os.getcwd() + '/'


    hardware_dict = None
    verbose = False
    bitlist = np.array([1, 2, 4, 5, 6, 7, 8])
    bitlist_cloud = np.array([16,32])
    for model_name in model_names[2:3]:

        input_stats_file = root_dir + 'generated/hw_simulator/post_process/' + model_name + '.csv'
        model_activation_stats = root_dir + '/tools/run_quantization/sample_yaml/'+model_name + '_quant_stats.yaml'


        args = parser.add_cmdline_args(classifier.init_classifier_compression_arg_parser(False)).parse_args()
        # parser = argparse.ArgumentParser(description='Data Loader')
        # add_post_train_quant_args(parser)
        # args = parser.parse_args()
        # args.device='cpu'
        # args.qe_config_file = False
        # args.qe_bits_acts = 0
        # args.qe_bits_wts = 0
        # args.qe_mode_acts= "sym"
        # args.qe_mode_wts= "sym"
        # args.qe_mode = "sym"

        # args.qe_no_quant_layers = None
        args.logdir = root_dir + 'generated/bit_search/quantizer/' + model_name + '/'
        Path(args.logdir).mkdir(parents=True, exist_ok=True)
        DataLoader(model_name, input_stats_file, bitlist, bitlist_cloud, model_activation_stats,args)


        random.seed(190087)







        # # activations
        # dnn_act_d = AttrDict()
        # dnn_act_d['vol'] = OrderedDict()
        # act_vol_df = df['ofm_vol']
        # total_act_vol = act_vol_df.sum()
        # dnn_act_d.layers_df = pd.DataFrame(index=layer_names_df, columns=['vol'])
        # dnn_act_d['sat_min'] = 0
        # dnn_act_d['sat_max'] = 6
        # for layer_name, act_vol in zip(layer_names_df,act_vol_df):
        #     # Assuming relu6
        #     max_act = random.uniform(0.7,6)
        #     dnn_act_d['vol'][layer_name] = np.random.uniform(0,max_act,act_vol)
        #     dnn_act_d.layers_df.loc[layer_name] = act_vol
        #
        # # run Memory Constraint Solver
        # # bitlist = np.array([1, 2, 4, 6, 8])

        # # DEBUG = {'split_idx': 8, 'act_constraint': 4, 'mem_constraint': 3.14}
        # ResNet50_memory_KB = 2**12
        # # Mobilenet_v2_memory_KB = 2**10
        # # DenseNet201_mem_KB = 2**13
        # # ResNet18_mem_KB = 2**10
        # BitSolver(model_name, df, dnn_act_d, dnn_weights_d, bitlist,bitlist_cloud, total_memory_size_KB=ResNet50_memory_KB,
        #           lambda_step_mul_const=2, lambda_step_div_const=15, is_auto_split=True)
        # print(datetime.datetime.now())

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
from ast import literal_eval
import re
import argparse


class DADS():

    def __init__(self,model_name, net_df, dnn_act_d, bitlist, total_memory_size_KB,
                 SELECTED_BIT=16, image_size=224*224*3*8):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.logger_auto = logging.getLogger('DADS')
        self.SELECTED_BIT=str(SELECTED_BIT)
        # Constants
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = 'generated/dads/' + model_name + '_' + self.timestamp  + '/'
        Path(filepath).mkdir(parents=True, exist_ok=True)
        self.result_dir = filepath
        self.stats_filename =  filepath +  '/latency_mse_stats.csv'

        self.hw_results_dir= 'generated/hw_results/' + model_name + '_' + self.timestamp  + '/'
        Path(filepath).mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.IMAGE_SIZE = image_size
        self.CLOUD_BITS = int(16)
        self.EDGE_MAX_BITS = int(16)
        # self.BW_LATENCY = 1/3/1024/1024 #  3 Mbits/sec
        # self.EDGE_TIME_PERIOD = 1.0/(202.3809524*(10**6)) # 202.381 MHz
        # self.CLOUD_TIME_PERIOD = 1.0/(1464.84375*(10**6)) # 1464.844 MHz

        # Objects
        logger_latency = logging.getLogger('LATENCY')
        self.net_df = copy.deepcopy(net_df)
        self.model_latency = ModelLatency(logger_latency, num_threads=16, data_dir_path=None, net_df=self.net_df)

        # Store net_df for latency calculation later
        self.net_df.to_csv(filepath +  '/net_df.csv', index=False, header=True)

        self.stats = pd.DataFrame(columns=['name', 'mse', 'latency','edge_sec', 'tr_sec', 'cloud_sec',
                                           'wgt_bits','act_bits', 'tr_bits', 'no_act_compr','act_opt',
                                           'split_idx', 'model_size_KB'])

        self.device_memory_KB = total_memory_size_KB

        # Functions
        self.create_split_point_table(dnn_act_d, bitlist)
        split_df = self.find_split_points()

        for idx, row in split_df.iterrows():
            layer_name = row['layer_name']
            layer_idx = row['layer_idx']
            self.split_dnn_and_generate_stats(layer_idx)

        # Edge config
        num_layers = len(dnn_act_d.layers_df)
        self.split_dnn_and_generate_stats(num_layers-1)
        # Cloud config
        self.split_dnn_and_generate_stats(-1)

    def split_dnn_and_generate_stats(self, layer_idx):

        edge_df, cloud_df = self.get_split_dnns(layer_idx)
        ifFits, model_size_KB = self.check_if_model_fits(edge_df)
        name = '{}_{}'.format(layer_idx,int(model_size_KB))
        self.logger_auto.info('--- {} ---'.format(name))

        if ifFits == False:
            self.logger_auto.warning('Model does not fit in edge device')


        self.stats = self.stats.append({'name': name,
                                        'split_idx': layer_idx,
                                        'model_size_KB': model_size_KB
                                        },
                                       ignore_index=True)
        # self.logger_auto.debug(self.stats)
        self.dump_bit_configurations(name, edge_df, cloud_df)

    def check_if_model_fits(self, edge_df):
        edge_weight_size_df = edge_df['wgt_vol'] * edge_df['bit_weights']
        edge_weight_size = edge_weight_size_df.sum()
        edge_act_size_df = edge_df['ofm_vol'] * edge_df['bit_activations']
        edge_act_size = max(self.IMAGE_SIZE, edge_act_size_df.max())
        edge_model_size_KB = (edge_weight_size + edge_act_size)/8/1024
        self.logger_auto.info('edge_weight_size_KB: {} edge_act_size_KB: {} '.format(edge_weight_size/8/1024,edge_act_size/8/1024 ))
        if edge_model_size_KB > self.device_memory_KB:
            self.logger_auto.info('-- does not fit on device memory -- ')
            return False, edge_model_size_KB
        else:
            return True, edge_model_size_KB

    def create_split_point_table(self, dnn_act_d, bitlist):
        split_point_table = []
        layer_names = list(dnn_act_d.layers_df.index)
        vols = list(dnn_act_d.layers_df['vol'])
        for vol in vols:
            bandwidth = [x*vol for x in bitlist]
            split_point_table.append(bandwidth)
        # self.logger_auto.debug('-- split points table: size(layer) - size(input)-- ')
        cols = [str(x) for x in bitlist]
        self.split_point_table = pd.DataFrame(split_point_table,index= layer_names, columns = cols)
        # get max act memory requirement
        # max_act_size = self.split_point_table[str(bitlist[-1])].max() + self.IMAGE_SIZE
        return # max_act_size/1024/8

    def get_required_bw_diff(self, layer_name, selected_bit):
        layer_stats = self.net_df[self.net_df['name'] == layer_name]
        layer_idx = layer_stats.index.tolist()[0]
        layer_rank = layer_stats['rank'].tolist()[0]
        required_bw = []
        required_bw.append(self.split_point_table.loc[layer_name][str(selected_bit)])

        # iterate over all cloud nodes and check if pred lies in the edge node.
        node_pred_list = []
        for node_idx in range(layer_idx + 1, len(self.net_df)):
            node_stats = self.net_df.loc[node_idx]
            node_pred_list.extend(node_stats['pred'])
            if node_pred_list is np.NaN:
                continue

        # remove duplicates
        node_pred_list = list(dict.fromkeys(node_pred_list))
        # select node dependencies
        node_pred_list = [x for x in node_pred_list if x< layer_rank]

        for pred_rank in node_pred_list:
            pred_stats = self.net_df[self.net_df['rank'] == pred_rank]
            pred_name = pred_stats['name'].to_list()[0]
            pred_act_bits = pred_stats['bit_activations'].to_list()[0]
            required_bw.append(self.split_point_table.loc[pred_name][str(pred_act_bits)])

        required_bw_diff = sum(required_bw) - self.IMAGE_SIZE
        return required_bw_diff, layer_idx

    def find_split_points(self):
        guess_split_points = self.split_point_table - self.IMAGE_SIZE
        guess_split_points = guess_split_points[self.SELECTED_BIT].sort_values()

        if model_name == 'resnet50':
            auto_split_bw_diff = guess_split_points['layer2.0.downsample.0']
            guess_split_points = guess_split_points[guess_split_points < 0]
            # guess_split_points.append([{'layer2.0.downsample.0': auto_split_bw_diff}], ignore_index=True)
            guess_split_points.loc['layer2.0.downsample.0'] = auto_split_bw_diff
        else:
            guess_split_points = guess_split_points[guess_split_points < 0]

        if guess_split_points.empty:
            raise Exception('No possible split points available at this bit-width = {}'.format(self.SELECTED_BIT))


        potential_split_points = []
        for layer_name, _ in guess_split_points.iteritems():
            required_bw_diff, layer_idx = self.get_required_bw_diff(layer_name, self.SELECTED_BIT)
            potential_split_points.append([layer_name, required_bw_diff, layer_idx])


        potential_split_points = sorted(potential_split_points, key=itemgetter(1,2))
        prev_val = None
        potential_split_points2 = []
        # Optimization: if three layers have the same bandwidth requirement,
        # then the one with lowest layer index is a better split point.
        # Since, tansmission cost is same and cloud is faster than edge DNN [Assumption].
        for item in potential_split_points:
            layer_name, bw_diff, layer_idx = item
            if prev_val is None:
                potential_split_points2.append([layer_name,bw_diff, layer_idx])
                prev_val = bw_diff
            else:
                if prev_val == bw_diff:
                    continue
                else:
                    potential_split_points2.append([layer_name, bw_diff,layer_idx])
                    prev_val = bw_diff




        df = pd.DataFrame(potential_split_points2, columns = ['layer_name','required_bw_diff', 'layer_idx'])
        dfidx = df[df['required_bw_diff'] >= 0].index
        num_positive_bw_diff = len(df[df['required_bw_diff'] >=0])
        len_df = len(df)
        if num_positive_bw_diff == len_df:
            self.logger_auto.error('Unlikely that a split solution exists'.format(df))
        else:
            if model_name == 'resnet50':
                auto_split_point =  df[df['layer_name'] == 'layer2.0.downsample.0']
                df.drop(dfidx, inplace=True)
                df = df.append(auto_split_point, ignore_index=True)
            else:
                df.drop(dfidx, inplace=True)
        self.logger_auto.info('-- selected split points --')
        self.logger_auto.info(df)
        return df

    def get_split_dnns(self, layer_idx):

        num_edge_layers = layer_idx + 1
        edge_df = self.net_df[0:num_edge_layers].copy()
        cloud_df = self.net_df[num_edge_layers:].copy()
        cloud_df.reset_index(inplace=True)

        return edge_df, cloud_df

    def dump_bit_configurations(self, name, edge_df, cloud_df):

        df1 = edge_df[['name', 'bit_weights', 'bit_activations']]
        df2 = cloud_df[['name', 'bit_weights', 'bit_activations']]
        df = df1.append(df2)
        df = df.astype({'bit_weights': int, 'bit_activations': int})

        bit_file_name=self.result_dir +'/' + name + '.csv'
        df.to_csv(bit_file_name, index=False, header=True)
        self.stats.to_csv(self.stats_filename, index=False, header=True)


def string_to_int_list(list_str_csv):
    if list_str_csv is None or list_str_csv is np.NaN:
        return  None
    # elif type(list_str_csv) != str:
    #     return [list_str_csv]
    else:
        # list_int = literal_eval(list_str_csv)
        list_str = re.split(r'[\[\]]', list_str_csv)[1].split(',')
        list_int = [int(i) for i in list_str]
    return list_int

def main(model_name,device_memory_KB, NUMLAYERS=None):
    input_file_name = root_dir + 'generated/hw_simulator/post_process2/' + model_name + '.csv'
    input_df = pd.read_csv(input_file_name, index_col=0)
    input_df.reset_index(inplace=True, drop=True)
    SELECTED_BIT=8
    input_df['bit_weights'] = SELECTED_BIT
    input_df['bit_activations'] = SELECTED_BIT
    random.seed(190087)
    if NUMLAYERS is None:
        NUMLAYERS = len(input_df)

    input_df = input_df[0:NUMLAYERS]
    input_df['pred'] = input_df['pred'].apply(string_to_int_list)


    print('No Logdir selected: Running Autosplit v1')
    # Fix type issues
    input_df = input_df.astype({'ifm_vol': int, 'ofm_vol': int,
                                'wgt_vol': int, "mac": int,
                                'bit_weights': int,
                                'bit_activations': int,
                                'rank': int})

    layer_names_df = input_df['name']

    # activations
    dnn_act_d = AttrDict()
    act_vol_df = input_df['ofm_vol']
    dnn_act_d.layers_df = pd.DataFrame(index=layer_names_df, columns=['vol'])

    input_image_shape= re.split('[()]', input_df['ifm'][0])[1:-1]
    input_image_shape= input_image_shape[0].split(', ')
    input_image_vol=  np.prod([int(x) for x in input_image_shape])
    input_image_vol_bits = input_image_vol*8

    for layer_name, act_vol in zip(layer_names_df, act_vol_df):
        dnn_act_d.layers_df.loc[layer_name] = {'vol': act_vol}


    # To Store required bw diff at different bitwidths to analyze potential split points
    bitlist = np.array([1, 2, 4, 5, 6, 7, 8, 16, 32])
    print(datetime.datetime.now())


    DADS(model_name, input_df, dnn_act_d, bitlist, device_memory_KB,
     SELECTED_BIT=SELECTED_BIT, image_size=input_image_vol_bits)

    print(datetime.datetime.now())

if __name__ == '__main__':

    print(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', type=lambda s: s.lower(),
                        help='add graph information to model stats dataframe')

    args = parser.parse_args()
    print('Running model: {}'.format(args.arch))
    sel_model_name=args.arch

    device_memory_KB_per_model = {
        'CONV2D': 2**10,
        'yolov3-416': 2**17, 'yolov3-512': 2**18, 'yolov3-608': 2**19,
        'yolov3-spp-416': 2**17, 'yolov3-spp-512': 2**18,'yolov3-spp-608': 2**19,
        'yolov3-tiny-416': 2 ** 15, 'yolov3-tiny-512': 2 ** 15, 'yolov3-tiny-608': 2 ** 15,
        'resnet18':2**10 , 'resnet_fpn_graph': 2**12,'resnet50': 2**12, 'mobilenet_v2': 2**10,
        'densenet201':2**13, 'mnasnet1_0': 2**12, 'googlenet': 2**13, 'resnext50_32x4d': 2**12,'vgg16':  2**12
    }


    if sel_model_name not in device_memory_KB_per_model.keys():
        raise ValueError('Current model is not selected. Select from: \n {}'.format(device_memory_KB_per_model.keys()))
    else:
        model_name = sel_model_name
    root_dir = os.getcwd() + '/'

    # -----
    verbose = False
    # num_layers=5
    num_layers=None
    device_memory_KB  = device_memory_KB_per_model[model_name]
    main(model_name,device_memory_KB, NUMLAYERS = num_layers)

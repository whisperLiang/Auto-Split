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
import glob



class NeuroSurgeon():

    def __init__(self,model_name, net_df, BW, result_dir, logger, image_size=224*224*3*8):
        # self.SELECTED_BIT=str(SELECTED_BIT)
        self.logger_auto = logger
        # Constants
        self.bw=BW
        self.filepath = result_dir

        # NOTE: It has to be same for both edge and cloud. Since, in neurosurgeon_2_cycles: It uses CLOUDBITS
        # to calculate edge, cloud, transmission
        # self.CLOUDBITS=16
        # self.EDGEBITS=16

        self.CLOUDBITS=8
        self.EDGEBITS=8

        # self.result_dir = filepath
        # self.stats_filename =  filepath +  '/latency_mse_stats.csv'
        # self.hw_results_dir= 'generated/hw_results/' + model_name + '_' + self.timestamp  + '/'
        # Path(self.filepath).mkdir(parents=True, exist_ok=True)

        self.model_name = model_name
        self.IMAGE_SIZE = image_size

        # Objects
        logger_latency = logging.getLogger('LATENCY')
        self.net_df = copy.deepcopy(net_df)
        del self.net_df['bit_weights']
        del self.net_df['bit_activations']

        self.model_latency = ModelLatency(logger_latency, num_threads=0, bw=self.bw, data_dir_path=None,
                                          net_df=self.net_df)

        self.add_cycles_to_net_df(device='cloud')
        self.add_cycles_to_net_df(device='edge')

        # Store net_df for latency calculation later
        self.net_df.to_csv(self.filepath +  '/net_df.csv', index=False, header=True)

        print()

    def remove_last_row(self, file_name):
        f = open(file_name, "r")
        lines = f.readlines()
        lines = lines[:-1]
        f.close()
        with open(file_name, 'w') as the_file:
            for line in lines:
                the_file.write(line)

        return

    def add_cycles_to_net_df(self, device):
        # -- get latency
        bit_df = copy.deepcopy(self.net_df)
        if device == 'edge':
            bit_df['bit_activations'] = self.EDGEBITS
            bit_df['bit_weights'] = self.EDGEBITS
        else:
            bit_df['bit_activations'] = self.CLOUDBITS
            bit_df['bit_weights'] = self.CLOUDBITS

        _ = self.model_latency.get_latency(bit_df,device,
                                           '16_-1_16_False_False_{}_{}'.format(self.EDGEBITS, self.CLOUDBITS),
                                           '{}/'.format(self.filepath))
        cloud_stats_file = glob.glob('{}/16_-1_16_False_False_{}_{}_{}/*.csv'.format(self.filepath,self.EDGEBITS,
                                                                                     self.CLOUDBITS,device))[0]
        if 'yolov3-tiny' in cloud_stats_file:
            self.remove_last_row(cloud_stats_file)
        cloud_stats_df = pd.read_csv(cloud_stats_file, index_col=0, header=None).T
        cloud_stats_df = cloud_stats_df.reset_index()
        device_cycles_df = cloud_stats_df['cycles_total']
        diff_cloud = len(self.net_df) - len(device_cycles_df)
        num_fc_layers = len(self.net_df[self.net_df['type'].str.contains('Linear')])
        assert num_fc_layers == diff_cloud, 'Num FC layers != diff_cloud_index'
        assert diff_cloud <=1, 'More than one FC layer in net_df'
        if num_fc_layers >0:
            device_cycles_df.append(pd.Series([0.0] * num_fc_layers))
        self.net_df['{}_cycles'.format(device)] = device_cycles_df
        return




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

def main(model_name, NUMLAYERS=None):
    input_file_name = root_dir + 'generated/hw_simulator/post_process2/' + model_name + '.csv'
    input_df = pd.read_csv(input_file_name, index_col=0)
    input_df.reset_index(inplace=True, drop=True)
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

    # layer_names_df = input_df['name']

    # activations
    # dnn_act_d = AttrDict()
    # act_vol_df = input_df['ofm_vol']
    # dnn_act_d.layers_df = pd.DataFrame(index=layer_names_df, columns=['vol'])

    input_image_shape = re.split('[()]', input_df['ifm'][0])[1:-1]
    input_image_shape = input_image_shape[0].split(', ')
    input_image_vol = np.prod([int(x) for x in input_image_shape])
    input_image_vol_bits = input_image_vol * 8

    # for layer_name, act_vol in zip(layer_names_df, act_vol_df):
    #     dnn_act_d.layers_df.loc[layer_name] = {'vol': act_vol}

    # To Store required bw diff at different bitwidths to analyze potential split points
    # bitlist = np.array([1, 2, 4, 5, 6, 7, 8, 16, 32])
    # print(datetime.datetime.now())

    timestamp = time.strftime("%Y%m%d-%H%M%S")
    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger = logging.getLogger('NeuroSurgeon')
    # for bw in [1, 3, 10, 20]:
    for bw in [3]:
        logger.info('===============================================')
        logger.info('Running Bandwidth: {}'.format(bw))
        logger.info('===============================================')
        filepath = os.path.join('generated', 'neurosurgeon', '{}_{}'.format(model_name,timestamp),
                                     str(bw))
        Path(filepath).mkdir(parents=True, exist_ok=True)


        NeuroSurgeon(model_name, input_df, bw, filepath, logger, image_size=input_image_vol_bits)
        print(datetime.datetime.now())

if __name__ == '__main__':

    FLOAT_BITS = 16
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', type=lambda s: s.lower(),
                        help='add graph information to model stats dataframe')

    args = parser.parse_args()
    print('Running model: {}'.format(args.arch))
    sel_model_name=args.arch
    model_name = sel_model_name
    root_dir = os.getcwd() + '/'
    # -----
    verbose = False
    # num_layers=5
    num_layers = None
    main(model_name, NUMLAYERS=num_layers)


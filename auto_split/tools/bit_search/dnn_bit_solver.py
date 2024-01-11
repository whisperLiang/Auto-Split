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


class BitSolver():

    def __init__(self,model_name, net_df, dnn_act_d, dnn_weights_d, bitlist, bitlist_cloud, total_memory_size_KB,
                                                      lambda_step_mul_const=2, lambda_step_div_const=15,
                                                      image_size=224*224*3*8, is_auto_split=True, DEBUG=None):
        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.logger_auto = logging.getLogger('Bit-Solver')

        self.logger_act = logging.getLogger('ACT')
        self.logger_weights = logging.getLogger('WGT')
        # Constants
        self.timestamp = time.strftime("%Y%m%d-%H%M%S")
        filepath = 'generated/bit_search/' + model_name + '_' + self.timestamp  + '/'
        Path(filepath).mkdir(parents=True, exist_ok=True)
        self.result_dir = filepath
        self.stats_filename =  filepath +  '/latency_mse_stats.csv'

        self.hw_results_dir= 'generated/hw_results/' + model_name + '_' + self.timestamp  + '/'
        Path(filepath).mkdir(parents=True, exist_ok=True)
        self.model_name = model_name
        self.IMAGE_SIZE = image_size
        self.CLOUD_BITS = int(16)
        self.EDGE_MAX_BITS = int(8)
        # self.BW_LATENCY = 1/3/1024/1024 #  3 Mbits/sec
        # self.EDGE_TIME_PERIOD = 1.0/(202.3809524*(10**6)) # 202.381 MHz
        # self.CLOUD_TIME_PERIOD = 1.0/(1464.84375*(10**6)) # 1464.844 MHz

        # FLags
        self.max_act_bits_optimization = True
        self.get_latency_stats = False

        # Objects
        logger_latency = logging.getLogger('LATENCY')
        self.model_latency = ModelLatency(logger_latency, num_threads=16, data_dir_path=None, net_df=self.net_df)
        self.net_df = copy.deepcopy(net_df)
        # Store net_df for latency calculation later
        self.net_df.to_csv(filepath +  '/net_df.csv', index=False, header=True)

        self.stats = pd.DataFrame(columns=['name', 'mse', 'latency','edge_sec', 'tr_sec', 'cloud_sec',
                                           'wgt_bits','act_bits', 'tr_bits', 'no_act_compr','act_opt',
                                           'split_idx', 'model_size_KB'])
        avg_bit_constraint = 4
        self.logger_auto.debug('Initializing weight_memm_solver')
        self.weight_memory_solver = MemorySolver(dnn_weights_d, avg_bit_constraint, bitlist, bitlist_cloud,
                                        lambda_step_mul_const,
                                        lambda_step_div_const, logger=self.logger_weights)
        self.logger_auto.debug('Initializing act_memm_solver')
        self.act_memory_solver = MemorySolver(dnn_act_d, avg_bit_constraint, bitlist, bitlist_cloud,
                                        lambda_step_mul_const,
                                        lambda_step_div_const, logger=self.logger_act)
        self.device_memory_KB = total_memory_size_KB

        # Run Solvers for weights and activations separately
        self.weight_memory_solver.mem_constraint_solver()
        # self.act_memory_solver.mem_constraint_solver()



if __name__ == '__main__':

    print(datetime.datetime.now())
    model_names = ['CONV2D', 'resnet18', 'resnet50', 'mobilenet_v2', 'densenet201']
    device = 'cloud'
    root_dir = os.getcwd() + '/'

    # -----
    if device == 'edge':
        hardware_yaml = root_dir + "/tools/hw_simulator/schedule/params/hardware_config_edge.yaml"
    elif device == 'cloud':
        hardware_yaml = root_dir + "/tools/hw_simulator/schedule/params/hardware_config_cloud.yaml"

    hardware_dict = None
    verbose = False
    for model_name in model_names[2:3]:
        input_file_name = root_dir + 'generated/hw_simulator/post_process/' + model_name + '.csv'
        df = pd.read_csv(input_file_name)
        random.seed(190087)
        NUMLAYERS = len(df)
        # NUMLAYERS = 5
        df = df[0:NUMLAYERS]
        dnn_weights_d = AttrDict()
        dnn_weights_d['vol'] = OrderedDict()

        layer_names_df = df['name']
        weight_vol_df = df['wgt_vol']
        total_wgt_vol = weight_vol_df.sum()
        dnn_weights_d.layers_df = pd.DataFrame(index=layer_names_df, columns = ['vol'])
        dnn_weights_d['sat_min'] = -1
        dnn_weights_d['sat_max'] = 1
        for layer_name, wgt_vol in zip(layer_names_df,weight_vol_df):
            dnn_weights_d['vol'][layer_name] = np.random.uniform(-1,1,wgt_vol)
            dnn_weights_d.layers_df.loc[layer_name] = wgt_vol

        # activations
        dnn_act_d = AttrDict()
        dnn_act_d['vol'] = OrderedDict()
        act_vol_df = df['ofm_vol']
        total_act_vol = act_vol_df.sum()
        dnn_act_d.layers_df = pd.DataFrame(index=layer_names_df, columns=['vol'])
        dnn_act_d['sat_min'] = 0
        dnn_act_d['sat_max'] = 6
        for layer_name, act_vol in zip(layer_names_df,act_vol_df):
            # Assuming relu6
            max_act = random.uniform(0.7,6)
            dnn_act_d['vol'][layer_name] = np.random.uniform(0,max_act,act_vol)
            dnn_act_d.layers_df.loc[layer_name] = act_vol

        # run Memory Constraint Solver
        # bitlist = np.array([1, 2, 4, 6, 8])
        bitlist = np.array([1, 2, 4, 5, 6, 7, 8])
        bitlist_cloud = np.array([16,32])
        # DEBUG = {'split_idx': 8, 'act_constraint': 4, 'mem_constraint': 3.14}
        ResNet50_memory_KB = 2**12
        # Mobilenet_v2_memory_KB = 2**10
        # DenseNet201_mem_KB = 2**13
        # ResNet18_mem_KB = 2**10
        BitSolver(model_name, df, dnn_act_d, dnn_weights_d, bitlist,bitlist_cloud, total_memory_size_KB=ResNet50_memory_KB,
                  lambda_step_mul_const=2, lambda_step_div_const=15, is_auto_split=True)
        print(datetime.datetime.now())

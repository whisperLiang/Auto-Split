import datetime
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
from multiprocessing import Pool
import subprocess
import multiprocessing
import argparse
import re
import glob
from pathlib import Path

class ModelLatency():
    def __init__(self, logger, num_threads=16, data_dir_path=None, net_df=None):
        # Declare constants
        self.IMAGE_SIZE =None
        self.CLOUD_BITS = int(16)
        self.EDGE_MAX_BITS = int(16)
        # Specific to Latency calculation
        self.BW_LATENCY = 1/3/1024/1024 #  3 Mbits/sec
        self.EDGE_TIME_PERIOD = 1.0/(202.3809524*(10**6)) # 202.381 MHz
        self.CLOUD_TIME_PERIOD = 1.0/(1464.84375*(10**6)) # 1464.844 MHz
        self.logger = logger
        self.num_threads = num_threads

        if net_df is None:
            # FOR PARALLEL LATENCY RUNS ONLY
            self.is_parallel = True
            self.input_data_dir = data_dir_path
            # Generate result dir
            self.timestamp = self.input_data_dir.split('/')[2]
            filepath = 'generated/dads_latency/' + self.timestamp  + '/'
            Path(filepath).mkdir(parents=True, exist_ok=True)
            self.result_dir = filepath
            # Load net_df
            self.net_df = pd.read_csv(self.input_data_dir + '/net_df.csv')



        elif data_dir_path is None:
            self.is_parallel = False
            self.input_data_dir = None
            self.result_dir = None
            self.net_df = copy.deepcopy(net_df)


    def run_latency_parallel(self):
        if self.is_parallel == False:
            raise AssertionError('Set the class in parallel mode')

        #load latency_mse_stats
        self.stats_df = pd.read_csv(self.input_data_dir + '/latency_mse_stats.csv')

        def remove_config_with_no_files(ip_stats_df, ip_dir_path):
            p = Path(ip_dir_path).glob('*')
            files = [x for x in p if x.is_file()]
            config_names = [os.path.basename(x).split('.csv')[0] for x in files]
            ip_stats_df = ip_stats_df[ip_stats_df.name.isin(config_names)]
            ip_stats_df =  ip_stats_df.reset_index()
            return ip_stats_df

        self.stats_df = remove_config_with_no_files(self.stats_df, self.input_data_dir)
        self.spawn_latency_jobs(self.stats_df)

    def spawn_latency_jobs(self, selected_config_df):
        if self.is_parallel == False:
            raise AssertionError('Set the class in parallel mode')

        pool = multiprocessing.Pool(self.num_threads)
        # total_sec, edge_sec, transmission_sec, cloud_sec = \
        pool.map(self.per_thread_latency_calculation, selected_config_df['name'])
        return

    def get_split_dnns(self, layer_idx, bit_df):
        num_edge_layers = layer_idx + 1
        edge_df = self.net_df[0:num_edge_layers].copy()
        cloud_df = self.net_df[num_edge_layers:].copy()
        cloud_df.reset_index(inplace=True)
        return edge_df, cloud_df

    def per_thread_latency_calculation(self, file_name):
        split_idx = int(file_name.split('_')[0])
        bit_df = pd.read_csv(self.input_data_dir + '/' + file_name + '.csv')
        edge_df, cloud_df = self.get_split_dnns(split_idx, bit_df)
        total_sec, edge_sec, transmission_sec, cloud_sec = \
            self.get_model_latency(edge_df, cloud_df, split_idx, file_name)
        latency_stats = pd.DataFrame(columns=['edge_sec', 'tr_sec', 'cloud_sec', 'split_idx', 'total_sec'])
        latency_stats = latency_stats.append({
            'edge_sec': edge_sec, 'tr_sec': transmission_sec, 'cloud_sec': cloud_sec,
            'split_idx': split_idx, 'total_sec': total_sec
        }, ignore_index=True)

        latency_stats.to_csv(self.result_dir + '/' + file_name + '.csv', index=False, header=True)
        return


    def get_latency(self, net_df, device, name):
        root_dir = os.getcwd() + '/'

        if device == 'edge':
            hardware_yaml = root_dir + "/tools/hw_simulator/schedule/params/hardware_config_edge.yaml"
        elif device == 'cloud':
            hardware_yaml = root_dir + "/tools/hw_simulator/schedule/params/hardware_config_cloud.yaml"
        else:
            raise ValueError('device == edge or cloud only')
        model_name = name +'_' + device

        verbose = False
        net = Net(net_df)


        # -- HWCF Schedule ---
        schedule = HWCFSchedule(net, model_name, self.result_dir, verbose, hardware_yaml=hardware_yaml,
                                hardware_dict=None)

        schedule.run_model()
        total_cycles = schedule.print_stats()
        self.logger.debug('total cycles: {}'.format(total_cycles))
        return total_cycles

    def get_model_latency(self, edge_df, cloud_df, layer_idx, name):
        self.logger.info(datetime.datetime.now())
        edge_cycles = self.get_latency(edge_df, 'edge', name)
        edge_sec = edge_cycles * self.EDGE_TIME_PERIOD
        # self.logger.info(datetime.datetime.now())
        cloud_cycles = self.get_latency(cloud_df, 'cloud', name)
        cloud_sec = cloud_cycles * self.CLOUD_TIME_PERIOD
        self.logger.info(datetime.datetime.now())
        # All on cloud
        if layer_idx == -1:
            transmission_sec = self.IMAGE_SIZE *self.BW_LATENCY
        else:
            # Note incase of edge -- need to transmit results to cloud.
            transmission_sec = self.net_df.loc[layer_idx]['ofm_vol'] * \
                               edge_df.at[layer_idx, 'bit_activations'] * self.BW_LATENCY
        total_sec = edge_sec + transmission_sec + cloud_sec
        self.logger.debug('total sec: {} , edge_sec:{} transmission_sec: {} cloud_sec: {}'.format(
            total_sec, edge_sec, transmission_sec, cloud_sec))
        return total_sec, edge_sec, transmission_sec , cloud_sec



if __name__ == '__main__':
    print(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('--net-df', '-n',default='resnet18_20200927-130044', type=lambda s: s.lower(),
                        help=' Needs timestamp folder generated by autosplit algorithm ')
    parser.add_argument('--threads', '-t',default='16', type=int,
                        help=' number of threads to run in parallel ')

    args = parser.parse_args()
    num_threads=args.threads
    print('Running model: {} --threads {}'.format(args.net_df,args.threads))
    datadir=args.net_df
    data_dir_path = 'generated/dads/' + datadir + '/'

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger = logging.getLogger('LATENCY')

    model_latency = ModelLatency(logger,num_threads=num_threads,data_dir_path=data_dir_path, net_df=None)
    input_image_shape= re.split('[()]', model_latency.net_df['ifm'][0])[1:-1]
    input_image_shape= input_image_shape[0].split(', ')
    model_latency.IMAGE_SIZE = 8*np.prod([int(x) for x in input_image_shape])

    model_latency.run_latency_parallel()
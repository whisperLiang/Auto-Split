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
    def __init__(self, logger, num_threads=16, bw =3, data_dir_path=None, net_df=None):
        # Declare constants
        self.IMAGE_SIZE =None
        self.CLOUD_BITS = int(16)
        # Specific to Latency calculation
        self.bw = bw
        self.BW_LATENCY = 1/bw/1024/1024 #  3 Mbits/sec
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
            # filepath = 'generated/latency/' + self.timestamp  + '/'
            filepath = 'generated/latency/{}/{}/'.format(self.timestamp,self.bw)
            Path(filepath).mkdir(parents=True, exist_ok=True)
            self.result_dir = filepath
            # Load net_df
            self.net_df = pd.read_csv(self.input_data_dir + '/net_df.csv')



        elif data_dir_path is None:
            self.is_parallel = False
            self.input_data_dir = None
            self.result_dir = None
            self.net_df = copy.deepcopy(net_df)



    def run_latency_parallel(self, lower_mse_than_bit):
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
        selected_config_df = self.select_configurations(lower_mse_than_bit=lower_mse_than_bit,
                                                        config_name='True_True')
        self.spawn_latency_jobs(selected_config_df)

        # selected_config_df = self.select_configurations(config_name='True_False')
        # self.spawn_latency_jobs(selected_config_df)

        selected_config_df = self.select_configurations(config_name='False_True')
        self.spawn_latency_jobs(selected_config_df)

        selected_config_df = self.select_configurations(lower_mse_than_bit=lower_mse_than_bit,
                                                        config_name='False_False')
        self.spawn_latency_jobs(selected_config_df)

    def spawn_latency_jobs(self, selected_config_df):
        if self.is_parallel == False:
            raise AssertionError('Set the class in parallel mode')

        pool = multiprocessing.Pool(self.num_threads)
        # total_sec, edge_sec, transmission_sec, cloud_sec = \
        pool.map(self.per_thread_latency_calculation, selected_config_df['name'])
        pool.close()
        pool.join()
        return

    def get_split_dnns(self, layer_idx, bit_df):
        num_edge_layers = layer_idx + 1
        edge_df = self.net_df[0:num_edge_layers].copy()
        cloud_df = self.net_df[num_edge_layers:].copy()
        cloud_df.reset_index(inplace=True)

        # Update Edge act, weight bits
        edge_df['bit_weights'] = bit_df[0:num_edge_layers]['bit_weights']
        edge_df['bit_activations'] = bit_df[0:num_edge_layers]['bit_activations']

        # Update CLOUD act, weight bits
        num_cloud_layers = len(self.net_df) - num_edge_layers
        cloud_layer_bits = [self.CLOUD_BITS] * num_cloud_layers
        cloud_df['bit_weights'] = cloud_layer_bits
        cloud_df['bit_activations'] = cloud_layer_bits

        return edge_df, cloud_df

    def per_thread_latency_calculation(self, file_name):
        bit_df = pd.read_csv(self.input_data_dir + '/' + file_name + '.csv')
        num_edge_layers = bit_df[bit_df['bit_activations'] == self.CLOUD_BITS].index

        if num_edge_layers.empty:
            num_edge_layers = len(self.net_df)
        else:
            num_edge_layers = num_edge_layers[0]

        split_idx = num_edge_layers - 1
        edge_df, cloud_df = self.get_split_dnns(split_idx, bit_df)
        total_sec, edge_sec, transmission_sec, cloud_sec = \
            self.get_model_latency(edge_df, cloud_df, split_idx, file_name)
        latency_stats = pd.DataFrame(columns=['edge_sec', 'tr_sec', 'cloud_sec', 'split_idx', 'total_sec'])
        latency_stats = latency_stats.append({
            'edge_sec': edge_sec, 'tr_sec': transmission_sec, 'cloud_sec': cloud_sec,
            'split_idx': split_idx, 'total_sec': total_sec
        }, ignore_index=True)

        latency_stats.to_csv(self.result_dir + '/' + file_name + '.csv', index=False, header=True)
        # return total_sec, edge_sec, transmission_sec, cloud_sec
        return

    def select_configurations(self, lower_mse_than_bit=None, config_name='True_True'):
        num_layers = len(self.net_df)
        config_df = None
        if config_name == 'True_True' or config_name == 'True_False' or config_name == 'False_True':
            if lower_mse_than_bit is None:
                config_df = self.stats_df[self.stats_df['name'].str.contains(config_name)]
            else:

                ubit = str(lower_mse_than_bit)
                # Get Uniform 6bit mse
                u6_filename = ubit+ '_' + str(num_layers - 1) + '_'+ ubit +'_False_False'
                if u6_filename in self.stats_df['name'].unique():
                    u6_mse = self.stats_df[self.stats_df['name'] == u6_filename]['mse'].reset_index()
                    u6_mse = u6_mse.loc[0]['mse']
                    selected_stats = self.stats_df[self.stats_df['mse'] < u6_mse]
                    config_df = selected_stats[selected_stats['name'].str.contains(config_name)]

                else:
                    raise ValueError('No Solution available with MSE lower than uniform 6 bits.')

        elif config_name == 'False_False':
            config_df = self.stats_df[self.stats_df['name'].str.contains(config_name)]
        else:
            raise ValueError('config_name == True_False or True_True or False_False')

        return config_df

    def get_latency(self, net_df, device, name, result_dir=None):
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
        if result_dir is None:
            result_dir = self.result_dir

        schedule = HWCFSchedule(net, model_name, result_dir, verbose, hardware_yaml=hardware_yaml,
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
    parser.add_argument('--net-df', '-n',default='resnet50_20200521-174448', type=lambda s: s.lower(),
                        help=' Needs timestamp folder generated by autosplit algorithm ')
    parser.add_argument('--threads', '-t',default='16', type=int,
                        help=' number of threads to run in parallel ')

    parser.add_argument('--min-mse', '-m',default=None, type=int,
                        help='  Only select mse less than m bit ')
    args = parser.parse_args()
    num_threads=args.threads
    lower_mse_than_bit = args.min_mse
    print('Running model: {} --threads {} --minmse {}'.format(args.net_df,args.threads, args.min_mse))
    datadir=args.net_df
    data_dir_path = 'generated/bit_search/' + datadir + '/'

    logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    logger = logging.getLogger('LATENCY')

    for bw in [1,3,10,20]:
        logger.info('===============================================')
        logger.info('Running Bandwidth: {}'.format(bw))
        logger.info('===============================================')
        model_latency = ModelLatency(logger,num_threads=num_threads,bw=bw, data_dir_path=data_dir_path, net_df=None)
        input_image_shape= re.split('[()]', model_latency.net_df['ifm'][0])[1:-1]
        input_image_shape= input_image_shape[0].split(', ')
        model_latency.IMAGE_SIZE = 8*np.prod([int(x) for x in input_image_shape])

        model_latency.run_latency_parallel(lower_mse_than_bit)
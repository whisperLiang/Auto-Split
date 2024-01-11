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

class NeuroCycles():
    def __init__(self, logger, time_stamp, bw, is_neuro=False):
        # ---------------------------------------------------------------------------------
        # Initialize containers
        # ---------------------------------------------------------------------------------
        self.BYTE=8
        self.logger = logger
        self.ROUNDING=6
        self.BW = bw
        # Assumes BW in Mbps
        self.BW_LATENCY = 1 / bw / 1024 / 1024  # 3 Mbits/sec
        self.EDGE_TIME_PERIOD = 1.0 / (202.3809524 * (10 ** 6))  # 202.381 MHz
        self.CLOUD_TIME_PERIOD = 1.0 / (1464.84375 * (10 ** 6))  # 1464.844 MHz

        input_dir = os.path.join(root_dir,'generated','neurosurgeon',time_stamp, str(bw))
        _, CLOUDBITS, _ = [x for x in os.listdir(input_dir) if 'False' in x][0].split('_False_False_')[-1].split('_')
        self.FLOAT_BITS = int(CLOUDBITS)

        input_file_name = os.path.join(input_dir, 'net_df.csv')
        input_df = pd.read_csv(input_file_name, index_col=0)
        input_df = input_df[~input_df['name'].str.contains('fc')]
        input_df.reset_index(inplace=True, drop=True)
        input_df['bit_weights'] = self.FLOAT_BITS
        input_df['bit_activations'] = self.FLOAT_BITS

        self.NUMLAYERS = len(input_df)
        self.IMAGE_SIZE = input_df['ifm_vol'][0]*self.BYTE
        self.input_df = input_df
        self.create_split_point_table(bitlist=[self.FLOAT_BITS])
        self.is_neuro = is_neuro
        if 'mobilenet_v2' in time_stamp:
            self.model_name = 'mobilenet_v2'
        else:
            self.model_name = time_stamp.split('_')[0]

        # if is_neuro:
        #     print('=======================================')
        #     print('{}: NEUROSURGEON RESULTS'.format(self.model_name))
        #     print('=======================================')
        # else:
        #     print('=======================================')
        #     print('{}: DADS RESULTS'.format(self.model_name))
        #     print('=======================================')

        self.result_dir = os.path.join(root_dir,'generated','neurosurgeon',time_stamp, str(bw),'results')
        Path(self.result_dir).mkdir(parents=True, exist_ok=True)
        #---------------------------------------------------------------------------------
        # ---------------------------------------------------------------------------------

    def get_latency_stats(self):
        if self.is_neuro:
            cycles_df = self.calculate_neuro_split()
            return self.get_neuro_latency_stats(cycles_df), self.FLOAT_BITS
        else:
            cycles_df = self.calculate_dads_split(time_stamp, bw)
            return self.get_dads_latency_stats(cycles_df), self.FLOAT_BITS

    def get_dads_latency_stats(self, cycles_df):
        latency_stats = pd.DataFrame(columns=['edge_sec', 'tr_sec', 'cloud_sec', 'split_idx', 'total_sec'])
        # 1. Get Cloud-Only case..
        cloud_row_idx = cycles_df[cycles_df['idx'] == -1].index.tolist()[0]
        latency_stats = latency_stats.append({
            'edge_sec': cycles_df.iloc[cloud_row_idx]['edge'], 'tr_sec': cycles_df.iloc[cloud_row_idx]['tr'],
            'cloud_sec': cycles_df.iloc[cloud_row_idx]['cloud'],
            'split_idx': -1, 'total_sec': cycles_df.iloc[cloud_row_idx]['total']
        }, ignore_index=True)

        # 2. add edge-only case
        edge_row_idx =  cycles_df[cycles_df['idx'] == self.NUMLAYERS-1].index.tolist()[0]
        latency_stats = latency_stats.append({
            'edge_sec': cycles_df.iloc[edge_row_idx]['edge'], 'tr_sec': cycles_df.iloc[edge_row_idx]['tr'],
            'cloud_sec': cycles_df.iloc[edge_row_idx]['cloud'],
            'split_idx': self.NUMLAYERS-1, 'total_sec': cycles_df.iloc[edge_row_idx]['total']
        }, ignore_index=True)


        # 3. Get NeuroSplit idx, then calculate real latency with all dependencies.
        min_cycle_df_idx = cycles_df['total'].idxmin()
        min_idx = int(cycles_df.iloc[min_cycle_df_idx]['idx'])
        split_idx = min_idx
        edge_sec = cycles_df.iloc[min_cycle_df_idx]['edge']
        transmission_sec = cycles_df.iloc[min_cycle_df_idx]['tr']
        cloud_sec = cycles_df.iloc[min_cycle_df_idx]['cloud']
        total_sec = cycles_df.iloc[min_cycle_df_idx]['total']

        latency_stats = latency_stats.append({
            'edge_sec': edge_sec, 'tr_sec': transmission_sec ,
            'cloud_sec': cloud_sec,
            'split_idx': split_idx, 'total_sec': total_sec
        }, ignore_index=True)

        # print('{},NEURO.{},{}'.format(self.model_name, self.split_idx, total_sec))
        latency_stats['Latency'] = latency_stats['total_sec'] / latency_stats['total_sec'].max()
        # print(latency_stats[['split_idx', 'Latency']])
        return latency_stats

    def get_neuro_latency_stats(self, cycles_df):
        latency_stats = pd.DataFrame(columns=['edge_sec', 'tr_sec', 'cloud_sec', 'split_idx', 'total_sec'])
        # 1. Get Cloud-Only case..
        cloud_row_idx = cycles_df[cycles_df['idx'] == -1].index.tolist()[0]
        latency_stats = latency_stats.append({
            'edge_sec': cycles_df.iloc[cloud_row_idx]['edge'], 'tr_sec': cycles_df.iloc[cloud_row_idx]['tr'],
            'cloud_sec': cycles_df.iloc[cloud_row_idx]['cloud'],
            'split_idx': -1, 'total_sec': cycles_df.iloc[cloud_row_idx]['total']
        }, ignore_index=True)

        # 2. add edge-only case
        edge_row_idx =  cycles_df[cycles_df['idx'] == self.NUMLAYERS-1].index.tolist()[0]
        latency_stats = latency_stats.append({
            'edge_sec': cycles_df.iloc[edge_row_idx]['edge'], 'tr_sec': cycles_df.iloc[edge_row_idx]['tr'],
            'cloud_sec': cycles_df.iloc[edge_row_idx]['cloud'],
            'split_idx': self.NUMLAYERS-1, 'total_sec': cycles_df.iloc[edge_row_idx]['total']
        }, ignore_index=True)


        # 3. Get NeuroSplit idx, then calculate real latency with all dependencies.
        min_cycle_df_idx = cycles_df['total'].idxmin()
        min_idx = int(cycles_df.iloc[min_cycle_df_idx]['idx'])
        #neuro_estimate which is wrong.
        orig_total_sec = cycles_df.iloc[min_cycle_df_idx]['total']
        self.split_idx = min_idx
        split_layer_name = self.input_df.iloc[self.split_idx]['name']
        actual_bw_bits, layer_idx = self.get_required_bw(self.input_df, split_layer_name, self.FLOAT_BITS)
        neuro_bw_bits= self.input_df.iloc[self.split_idx]['ofm_vol']*self.FLOAT_BITS
        bw_diff_bits = actual_bw_bits - neuro_bw_bits
        # print('Diff_actual-neuro-bw: {}'.format(bw_diff_bits))
        edge_sec = cycles_df.iloc[min_cycle_df_idx]['edge']
        transmission_sec = actual_bw_bits*self.BW_LATENCY
        cloud_sec = cycles_df.iloc[min_cycle_df_idx]['cloud']
        total_sec = edge_sec + transmission_sec + cloud_sec

        latency_stats = latency_stats.append({
            'edge_sec': edge_sec, 'tr_sec': transmission_sec ,
            'cloud_sec': cloud_sec,
            'split_idx': self.split_idx, 'total_sec': total_sec
        }, ignore_index=True)

        # print('{},NEURO.{},{}'.format(self.model_name, self.split_idx, total_sec))
        latency_stats['Latency'] = latency_stats['total_sec'] / latency_stats['total_sec'].max()
        # print(latency_stats[['split_idx', 'Latency']])
        return latency_stats

    def run_latency(self, split_index):
        edge_df, cloud_df = self.get_split_dnns(split_index)
        total_sec, edge_sec, transmission_sec, cloud_sec = \
            self.get_model_latency(edge_df, cloud_df, split_index, '{}.csv'.format(split_index))
        self.latency_stats = self.latency_stats.append({
            'edge_sec': edge_sec, 'tr_sec': transmission_sec, 'cloud_sec': cloud_sec,
            'split_idx': split_index, 'total_sec': total_sec
        }, ignore_index=True)
        return

    def get_model_latency(self, edge_df, cloud_df, layer_idx, name):
        # self.logger.info(datetime.datetime.now())
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

    def get_split_dnns(self, layer_idx):
        num_edge_layers = layer_idx + 1
        edge_df = self.net_df[0:num_edge_layers].copy()
        cloud_df = self.net_df[num_edge_layers:].copy()
        cloud_df.reset_index(inplace=True)
        return edge_df, cloud_df

    def create_split_point_table(self, bitlist):
        split_point_table = []
        layer_names = list(self.input_df['name'])
        vols = list(self.input_df['ofm_vol'])
        for vol in vols:
            bandwidth = [x*vol for x in bitlist]
            split_point_table.append(bandwidth)
        # self.logger_auto.debug('-- split points table: size(layer) - size(input)-- ')
        cols = [str(x) for x in bitlist]
        self.split_point_table = pd.DataFrame(split_point_table,index= layer_names, columns = cols)
        return

    def get_required_bw(self, net_df, layer_name, layer_act_bit):
        layer_stats = net_df[net_df['name'] == layer_name]
        layer_idx = layer_stats.index.tolist()[0]
        layer_rank = layer_stats['rank'].tolist()[0]
        required_bw = []
        required_bw.append(self.split_point_table.loc[layer_name][str(layer_act_bit)])

        # iterate over all cloud nodes and check if pred lies in the edge node.
        node_pred_list = []
        for node_idx in range(layer_idx + 1, len(net_df)):
            node_stats = net_df.loc[node_idx]
            node_pred_list.extend(literal_eval(node_stats['pred']))
            if node_pred_list is np.NaN:
                continue

        # remove duplicates
        node_pred_list = list(dict.fromkeys(node_pred_list))
        # select node dependencies
        node_pred_list = [x for x in node_pred_list if x < layer_rank]

        for pred_rank in node_pred_list:
            pred_stats = net_df[net_df['rank'] == pred_rank]
            pred_name = pred_stats['name'].to_list()[0]
            pred_act_bits = pred_stats['bit_activations'].to_list()[0]
            required_bw.append(self.split_point_table.loc[pred_name][str(pred_act_bits)])

        required_bw_val = sum(required_bw)
        return required_bw_val, layer_idx

    def calculate_neuro_split(self):
        cycles_df = pd.DataFrame(columns=['idx','edge','tr','cloud','total'])
        tr_0 = round(self.IMAGE_SIZE*self.BW_LATENCY,self.ROUNDING)
        cloud_0 = round(self.input_df['cloud_cycles'].sum()*self.CLOUD_TIME_PERIOD,self.ROUNDING)
        total_cycles = tr_0 + cloud_0
        cycles_df = cycles_df.append({'idx':-1, 'edge': 0,'tr':tr_0,'cloud': cloud_0, 'total': total_cycles}, ignore_index=True)

        for layer_idx in range(self.NUMLAYERS):
            edge_cycles = round(self.input_df['edge_cycles'][0:layer_idx+1].sum()*self.EDGE_TIME_PERIOD,self.ROUNDING)
            cloud_cycles = round(self.input_df['cloud_cycles'][layer_idx+1:self.NUMLAYERS].sum()*self.CLOUD_TIME_PERIOD,
                                 self.ROUNDING)
            transmission_cost = round(self.input_df['ofm_vol'][layer_idx]*self.FLOAT_BITS*self.BW_LATENCY,self.ROUNDING)
            # if layer_idx == NUMLAYERS-1:
            #     total_cycles = round(edge_cycles,4)
            #     cycles_df = cycles_df.append({'idx':layer_idx, 'edge': edge_cycles, 'tr':0, 'cloud': 0, 'total': total_cycles}, ignore_index=True)
            # else:
            total_cycles = round(edge_cycles + transmission_cost + cloud_cycles,self.ROUNDING)
            cycles_df = cycles_df.append({'idx':layer_idx, 'edge': edge_cycles, 'tr':transmission_cost,
                             'cloud': cloud_cycles, 'total': total_cycles},
                             ignore_index=True)


        return cycles_df

    def calculate_dads_split(self, time_stamp, bw):
        cycles_df = pd.DataFrame(columns=['idx','edge','tr','cloud','total'])
        tr_0 = round(self.IMAGE_SIZE*self.BW_LATENCY,self.ROUNDING)
        cloud_0 = round(self.input_df['cloud_cycles'].sum()*self.CLOUD_TIME_PERIOD,self.ROUNDING)
        total_cycles = tr_0 + cloud_0
        cycles_df = cycles_df.append({'idx':-1, 'edge': 0,'tr':tr_0,'cloud': cloud_0, 'total': total_cycles}, ignore_index=True)

        for layer_idx in range(self.NUMLAYERS):
            edge_cycles = round(self.input_df['edge_cycles'][0:layer_idx+1].sum()*self.EDGE_TIME_PERIOD,self.ROUNDING)
            cloud_cycles = round(self.input_df['cloud_cycles'][layer_idx+1:self.NUMLAYERS].sum()*self.CLOUD_TIME_PERIOD,
                                 self.ROUNDING)

            split_layer_name = self.input_df.iloc[layer_idx]['name']
            actual_bw_bits, layer_idx = self.get_required_bw(self.input_df, split_layer_name, self.FLOAT_BITS)

            transmission_cost = actual_bw_bits*self.BW_LATENCY
            total_cycles = edge_cycles + transmission_cost + cloud_cycles
            cycles_df = cycles_df.append({'idx':layer_idx, 'edge': edge_cycles, 'tr':transmission_cost,
                             'cloud': cloud_cycles, 'total': total_cycles},
                             ignore_index=True)


        return cycles_df


def collect_stats(model_name, latency_df, neuro_latency_df, dads_latency_df, FLOAT_BITS):
    ROUNDING = 6
    cloud16_sec = neuro_latency_df.iloc[0]['total_sec']
    u8_sec = neuro_latency_df.iloc[1]['total_sec']
    neuro_sec = neuro_latency_df.iloc[2]['total_sec']
    neuro_split_idx = int(neuro_latency_df.iloc[2]['split_idx'])
    if neuro_split_idx == -1:
        neuro_split_idx = 'CLOUD'

    dads_sec = dads_latency_df.iloc[2]['total_sec']
    dads_split_idx = int(dads_latency_df.iloc[2]['split_idx'])
    if dads_split_idx == -1:
        dads_split_idx = 'CLOUD'

    # Normalize
    max_sec = max(cloud16_sec, u8_sec, neuro_sec, dads_sec)
    cloud16_sec_n = round(cloud16_sec / max_sec, ROUNDING)
    u8_sec_n = round(u8_sec / max_sec, ROUNDING)
    neuro_sec_n = round(neuro_sec / max_sec, ROUNDING)
    dads_sec_n = round(dads_sec / max_sec, ROUNDING)

    latency_df = latency_df.append({'bench': model_name, 'name': 'auto-split',
                                    'Latency': 0.0, 'total_sec': 0.0}, ignore_index=True)
    latency_df = latency_df.append({'bench': model_name, 'name': 'NEURO.{}'.format(neuro_split_idx),
                                    'Latency': neuro_sec_n, 'total_sec': neuro_sec}, ignore_index=True)

    latency_df = latency_df.append({'bench': model_name, 'name': 'QDMP.{}'.format(dads_split_idx),
                                    'Latency': dads_sec_n, 'total_sec': dads_sec}, ignore_index=True)

    latency_df = latency_df.append({'bench': model_name, 'name': 'U{}'.format(FLOAT_BITS), 'Latency': u8_sec_n,
                                    'total_sec': u8_sec},
                                   ignore_index=True)

    latency_df = latency_df.append({'bench': model_name, 'name': 'CLOUD{}'.format(FLOAT_BITS), 'Latency': cloud16_sec_n,
                                    'total_sec': cloud16_sec},
                                   ignore_index=True)
    return latency_df

if __name__ == '__main__':

    print(datetime.datetime.now())
    # parser = argparse.ArgumentParser()
    # parser.add_argument('--timestamp', '-a', metavar='ARCH', default='resnet50', type=lambda s: s.lower(),
    #                     help='add graph information to model stats dataframe')
    #
    # args = parser.parse_args()
    # print('Running model: {}'.format(args.arch))
    # sel_model_name=args.arch
    # model_name = sel_model_name
    root_dir = os.getcwd() + '/'
    # -----
    verbose = False
    # num_layers=5
    num_layers = None
    # benchmarks = ['googlenet_20210201-021516', 'mobilenet_v2_20210201-021528',
    #               'resnet50_20210201-021414', 'yolov3-416_20210201-021940',
    #               'yolov3-tiny-416_20210201-022150', 'mnasnet1_0_20210201-021725',
    #               'resnet18_20210201-021406', 'resnext50_32x4d_20210201-021538',
    #               'yolov3-spp-416_20210201-021735']
    input_dir_path = os.path.join(root_dir, 'generated', 'neurosurgeon')

    benchmarks = []
    for name in os.listdir(input_dir_path):
        if os.path.isdir(os.path.join(input_dir_path,name)):
            benchmarks.append(name)

    logger = logging.getLogger('LATENCY')
    latency_df = pd.DataFrame(columns=['bench','name','Latency'])
    float_bits = -1
    for time_stamp in benchmarks:
        if 'mobilenet_v2' in time_stamp:
            model_name = 'mobilenet_v2'
        else:
            model_name = time_stamp.split('_')[0]

        # for bw in [1,3,10,20]:
        bw=3
        Neuro = NeuroCycles(logger, time_stamp, bw, is_neuro=True)
        neuro_latency_df, FLOAT_BITS = Neuro.get_latency_stats()
        Dads = NeuroCycles(logger, time_stamp, bw, is_neuro=False)
        dads_latency_df, _ = Dads.get_latency_stats()

        latency_df = collect_stats(model_name, latency_df, neuro_latency_df, dads_latency_df, FLOAT_BITS)
        float_bits = FLOAT_BITS
    latency_df.to_csv('generated/neurosurgeon/cycles_summary_{}.csv'.format(float_bits))
    print()
    # main(benchmarks[0],3)



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


class AutoSplit():

    def __init__(self,model_name, net_df, dnn_act_d, dnn_weights_d, bitlist, bitlist_cloud, total_memory_size_KB,
                                                      lambda_step_mul_const=2, lambda_step_div_const=15,
                                                      image_size=224*224*3*8, is_auto_split=True, DEBUG_SPLIT=None, DEBUG=None):

        logging.basicConfig(stream=sys.stderr, level=logging.INFO)
        self.logger_auto = logging.getLogger('AUTO-SPLIT')

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
        self.net_df = copy.deepcopy(net_df)
        self.model_latency = ModelLatency(logger_latency, num_threads=16, data_dir_path=None, net_df=self.net_df)

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

        # Functions
        self.create_split_point_table(dnn_act_d, bitlist)
        if is_auto_split:
            self.find_auto_split(DEBUG_SPLIT)
        # else:
        #     if DEBUG is None:
        #         raise ValueError('Provide DEBUG dict with keys = {split_idx: 8, act_constraint: 4, mem_constraint: 3.14}')
        #
        #     # Selected configuration:
        #     split_idx = DEBUG['split_idx']
        #     act_constraint = DEBUG['act_constraint']
        #     mem_constraint = DEBUG['mem_constraint']
        #     # split_idx = 8
        #     # act_constraint = 4
        #     # mem_constraint = 3.14
        #     self.selected_split_bit_config(act_constraint, split_idx, mem_constraint)

    def maximize_act_bits(self, act_layerwise_bits_df, layer_idx):
        act_bits_df = pd.DataFrame(index= act_layerwise_bits_df.index , columns=['bits'])
        orig_act_mem_df = act_layerwise_bits_df[0:layer_idx+1]['bits']*act_layerwise_bits_df[0:layer_idx+1]['vol']
        max_mem_bits = orig_act_mem_df.max()
        max_mem_idx = orig_act_mem_df[orig_act_mem_df == max_mem_bits].index
        # select max between input activation of image vs other activation layers.
        max_mem_bits = max(self.IMAGE_SIZE, max_mem_bits)

        # test = pd.DataFrame(columns=['before','after'])
        # test['before'] = act_layerwise_bits_df['bits']
        for idx, row in act_layerwise_bits_df[0:layer_idx].iterrows():
            bits, vol = row
            bitlist = self.act_memory_solver.bitlist
            vol_list = [ x*vol for x in bitlist]
            selected_vol = max(filter(lambda x: x <= max_mem_bits, vol_list))
            selected_vol_index = vol_list.index(selected_vol)
            selected_bits = bitlist[selected_vol_index]
            act_bits_df.loc[idx]['bits'] = selected_bits
        # copy rest of the bits
        x1 = list(act_bits_df[:layer_idx]['bits'])
        x2 = list(act_layerwise_bits_df[layer_idx:]['bits'])
        x1.extend(x2)
        act_bits_df['bits'] = x1
        # test['after'] = act_bits_df['bits']
        return act_bits_df

    def get_split_points(self):
        # start with 2 bit act memory and get split points
        for act_avg_bit_constraint in [2,4,6]:
            available_mem_KB, act_layerwise_bits_df, split_points, _, max_mem_bits,_ = \
                self.bitsolver_act(act_avg_bit_constraint,split_points=True)
            if split_points is not None:
                return max_mem_bits, act_layerwise_bits_df, split_points
            else:
                continue
        self.logger_auto.debug(self.stats)
        self.stats.to_csv(self.stats_filename, index=False, header=True)
        raise ValueError('No valid split point found. Either Memory too small or Run Everything on edge or cloud')

    def get_split_points_2(self):

        # selected_bitlist = table_entries.loc[0]['layerwise_selected_bits']
        potential_split_points = []
        for layer_name in self.split_point_table.index:
            bit=2
            required_bw_diff, layer_idx = self.get_required_bw_diff(layer_name, bit)
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
            self.logger_auto.warning('Unlikely that a split solution exists')
            self.logger_auto.warning('selecting top 5 solutions')
            num_selected = min(5,len_df)
            df = df.sort_values(by=['required_bw_diff', 'layer_idx'], ascending=True)
            df = df = df[0:num_selected]
        else:
            df.drop(dfidx, inplace=True)
        self.logger_auto.info('-- selected split points --')
        self.logger_auto.info(df)
        return df

    def find_auto_split(self, DEBUG_SPLIT=None):

        if DEBUG_SPLIT:
             split_points = pd.DataFrame(columns=['layer_index'])
             split_points['layer_index'] = DEBUG_SPLIT
        else:
            _, _, split_points = self.get_split_points()
            # split_points = self.get_split_points_2()


        for _, row in split_points.iterrows():
            if DEBUG_SPLIT:
                layer_idx = row[0]
            else:
                _, _, layer_idx = row

            for act_avg_bit_constraint in [2,4,6,8]:
                self.logger_auto.info('#################')
                self.logger_auto.info('selected_layer_idx: {}, selected avg bit constraint: {}'.format(
                    layer_idx,act_avg_bit_constraint))
                available_mem_KB, act_layerwise_bits_df, _, transmitted_act_bit, _, valid_split = \
                    self.bitsolver_act(act_avg_bit_constraint,split_idx = layer_idx, split_points=False)

                if available_mem_KB is None:
                    self.logger_auto.warning('Memory too low')
                    break

                self.logger_auto.info('valid_split: {}, available KB: {}'.format(valid_split, available_mem_KB))
                self.logger_auto.info('#################')
                if valid_split == False:
                    # No point increasing activation bits
                    # for the same split point. Hence, break the inner for loop
                    self.logger_auto.warning('invalid split point: IMAGE SIZE < transmission act size')
                    break


                # Since, edge_df = available memory. This available_mem has no significance. We
                # are better off using average_bit_constraint instead for bit solver.
                # available_mem_KB is useful to check after edge_df to see if device fits?

                _, wgt_layerwise_bits, weight_selected_mem_bits = \
                    self.bitsolver_weights(available_mem_KB,layer_idx)

                if wgt_layerwise_bits is None:
                    self.logger_auto.warning('Memory too low')
                    break

                self.logger_auto.info('selected_layer_idx: {}'.format(layer_idx))
                # updated act_layerwise_bits_df
                act_bits_act_opt_df = self.maximize_act_bits(act_layerwise_bits_df, layer_idx)
                act_bits_no_opt_df = act_layerwise_bits_df

                # all act = 8 bits
                self.split_dnn_and_generate_stats(weight_selected_mem_bits, act_avg_bit_constraint, layer_idx,
                                                  wgt_layerwise_bits, list(act_bits_act_opt_df['bits']),transmitted_act_bit,
                                                  is_act_compression=False, ismax_act_opt=True,
                                                  ignore_device_memory=True)
                # w/ maximize_act_bits
                self.split_dnn_and_generate_stats(weight_selected_mem_bits, act_avg_bit_constraint, layer_idx,
                                                  wgt_layerwise_bits, list(act_bits_act_opt_df['bits']),transmitted_act_bit,
                                                  is_act_compression=True, ismax_act_opt=True)

                # w/o any act optimization
                self.split_dnn_and_generate_stats(weight_selected_mem_bits, act_avg_bit_constraint, layer_idx,
                                                  wgt_layerwise_bits, list(act_bits_no_opt_df['bits']),transmitted_act_bit,
                                                  is_act_compression=True, ismax_act_opt=False)

                self.stats.to_csv(self.stats_filename, index=False, header=True)
            # end_for act_bit_constraint
        # end for split_idx
        self.logger_auto.debug(self.stats)

        # ALL one edge
        self.gen_stats_all_on_edge()

        # All on cloud runs only once - Since, all 8 bits
        self.gen_stats_all_on_cloud()
        return

    # def selected_split_bit_config(self, act_avg_bit_constraint, selected_split_idx=None):
    #
    #     # For loop over total_mem_constraint of B* = {2,4,6}:
    #     # Use mem solver for 2bits avg and get table_entry {R*, B*, MSE}
    #     # Bit solver for activations
    #     available_mem_KB, act_layerwise_bits_df, split_points, _, _,_ = \
    #         self.bitsolver_act(act_avg_bit_constraint)
    #     if available_mem_KB is None:
    #         self.logger_auto.warning('Memory too low')
    #         raise ValueError('Memory too low')
    #
    #     return

    def split_dnn_and_generate_stats(self, weight_selected_mem_bits, act_avg_bit_constraint, layer_idx,
                                     wgt_layerwise_bits, act_layerwise_bits,transmitted_act_bit,
                                     is_act_compression=True, ismax_act_opt=True, ignore_device_memory=False):

        edge_df, cloud_df = self.get_split_dnns(layer_idx, wgt_layerwise_bits, act_layerwise_bits,
                                                is_act_compression)
        weight_table_idx=0 # 1 == all lowest bits, 2 = all 8 bits
        total_mse = self.get_mse(edge_df, cloud_df)
        name = '{}_{}_{}_{}_{}'.format(weight_selected_mem_bits[weight_table_idx], layer_idx,
                                    act_avg_bit_constraint,is_act_compression, ismax_act_opt)
        self.logger_auto.info('--- {} ---'.format(name))
        ifFits, model_size_KB = self.check_if_model_fits(edge_df)

        #  Uniform 2,4,6 on edge, and all on cloud models are discounted
        if ifFits == False and (ignore_device_memory == False):
            self.logger_auto.error('Model does not fit in edge device')
            raise ValueError('Model does not fit in edge device')

        if self.get_latency_stats:
            runtime_sec, edge_sec, transmission_sec, cloud_sec \
                = self.model_latency.get_model_latency(edge_df, cloud_df, layer_idx, name)
            self.logger_auto.debug('total_sec: {} edge_sec: {} tr_sec: {}'.format(runtime_sec,edge_sec,transmission_sec))
        else:
            runtime_sec = None
            edge_sec = None
            transmission_sec = None
            cloud_sec = None

        self.stats = self.stats.append({'name': name, 'mse': total_mse,
                                        'accuracy': None,
                                        'wgt_bits': weight_selected_mem_bits[weight_table_idx],
                                        'split_idx': layer_idx, 'model_size_KB': model_size_KB,'latency':runtime_sec,
                                        'edge_sec': edge_sec, 'tr_sec': transmission_sec, 'cloud_sec':cloud_sec,
                                        'act_bits':act_avg_bit_constraint, 'tr_bits': transmitted_act_bit,
                                        'no_act_compr': is_act_compression,
                                        'act_opt': ismax_act_opt
                                        },
                                       ignore_index=True)
        # self.logger_auto.debug(self.stats)
        self.dump_bit_configurations(name, edge_df, cloud_df)

    def bitsolver_act(self, act_avg_bit_constraint, split_idx = None, split_points= False):

        if split_idx is None:
            self.act_memory_solver.reset_dnn_state(self.act_memory_solver.orig_dnn.num_layers, act_avg_bit_constraint,
                                               isKB=False)
        else:
            self.act_memory_solver.reset_dnn_state(split_idx+1, act_avg_bit_constraint,
                                               isKB=False)

        act_table_entries = self.act_memory_solver.mem_constraint_solver()
        if act_table_entries is None:
            # raise ValueError('activation memory too small: increase total memory')
            return None, None, None, None, None, None

        act_layerwise_bits_d = act_table_entries['layerwise_selected_bits'][0]
        act_layerwise_bits = [value for key, value in act_layerwise_bits_d.items()]
        # Get transmitted activation bits
        transmitted_act_bit = act_layerwise_bits[-1]
        act_layerwise_bits_df = pd.DataFrame(act_layerwise_bits,
                                             index=self.act_memory_solver.dnn_state_D.selected_dnn.layers_df.index, columns=['vol'])
        act_mem_df = act_layerwise_bits_df.mul(self.act_memory_solver.dnn_state_D.selected_dnn.layers_df)
        act_layerwise_bits_df.columns = ['bits']

        max_act_mem = act_mem_df['vol'].max()
        if self.IMAGE_SIZE > act_mem_df['vol'][-1]:
            valid_split = True
        else:
            valid_split  = False
        max_mem_bits = max(self.IMAGE_SIZE, max_act_mem)
        max_act_mem_KB = max_mem_bits / 1024 / 8


        available_mem_KB = self.device_memory_KB - max_act_mem_KB

        act_layerwise_bits_df['vol'] = self.act_memory_solver.dnn_state_D.selected_dnn.layers_df['vol']


        # Find split points sorted in order of bandwidth difference (low is better)
        if split_points == True:
            split_points = self.find_split_points(act_table_entries)


        return available_mem_KB, act_layerwise_bits_df, split_points, transmitted_act_bit, max_mem_bits, valid_split

    def bitsolver_weights(self, available_mem_KB, split_idx=None):

        if split_idx is None:
            num_layers = self.weight_memory_solver.orig_dnn.num_layers
        else:
            num_layers = split_idx + 1

        # total_wgt_vol= self.weight_memory_solver.orig_dnn.layers_df.iloc[0:num_layers].layers_df['vol'].sum()
        # mem_constraint = (self.device_memory_KB*1024*8 - max_act_mem_bits)/(1024*8)
        # RESET DNN based on the split index
        self.weight_memory_solver.reset_dnn_state(num_layers, available_mem_KB, isKB=True)


        weight_table_entries = self.weight_memory_solver.mem_constraint_solver()
        if weight_table_entries is None:
            # raise ValueError('activation memory too small: increase total memory')
            return None, None, None

        # Get weight bits
        table_idx = 0
        wgt_layerwise_bits_d = weight_table_entries['layerwise_selected_bits'][table_idx]
        weight_layerwise_bits = [value for key, value in wgt_layerwise_bits_d.items()]
        weight_selected_mem_bits = [round(x,2) for x in weight_table_entries['mem_bits']]
        return  weight_table_entries, weight_layerwise_bits, weight_selected_mem_bits

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


    def get_mse(self, edge_df, cloud_df):
        cloud_weight_mse = self.weight_memory_solver.get_mse(cloud_df[['name', 'bit_weights']])
        cloud_act_mse = self.act_memory_solver.get_mse(cloud_df[['name', 'bit_activations']])
        cloud_total_mse = cloud_weight_mse + cloud_act_mse

        edge_weight_mse = self.weight_memory_solver.get_mse(edge_df[['name', 'bit_weights']])
        edge_act_mse = self.act_memory_solver.get_mse(edge_df[['name', 'bit_activations']])
        edge_total_mse = edge_weight_mse + edge_act_mse
        return cloud_total_mse + edge_total_mse

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

    def get_required_bw_diff(self, layer_name, layer_act_bit):
        layer_stats = self.net_df[self.net_df['name'] == layer_name]
        layer_idx = layer_stats.index.tolist()[0]
        layer_rank = layer_stats['rank'].tolist()[0]
        required_bw = []
        required_bw.append(self.split_point_table.loc[layer_name][str(layer_act_bit)])

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


    def find_split_points(self, table_entries):

        selected_bitlist = table_entries.loc[0]['layerwise_selected_bits']
        potential_split_points = []
        for layer_name, bit in selected_bitlist.items():
            required_bw_diff, layer_idx = self.get_required_bw_diff(layer_name, bit)
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
            self.logger_auto.warning('Unlikely that a split solution exists')
            self.logger_auto.warning('selecting top 5 solutions')
            num_selected = min(5,len_df)
            df = df.sort_values(by=['required_bw_diff', 'layer_idx'], ascending=True)
            df = df = df[0:num_selected]
        else:
            df.drop(dfidx, inplace=True)
        self.logger_auto.info('-- selected split points --')
        self.logger_auto.info(df)
        return df

    def get_split_dnns(self, layer_idx, wgt_layerwise_bits, act_layerwise_bits, is_act_compression):

        num_edge_layers = layer_idx + 1
        edge_df = self.net_df[0:num_edge_layers].copy()
        cloud_df = self.net_df[num_edge_layers:].copy()
        cloud_df.reset_index(inplace=True)

        # Update weight bits - edge
        edge_weight_bits = wgt_layerwise_bits[0:num_edge_layers]
        edge_df['bit_weights'] = edge_weight_bits

        # # Update act bits - edge
        edge_act_bits = act_layerwise_bits[0:num_edge_layers]
        edge_df['bit_activations'] = edge_act_bits
        if is_act_compression == False:
            edge_act_bits_no_compression = [self.EDGE_MAX_BITS] * num_edge_layers
            # Transmission bit remains compressed
            edge_act_bits_no_compression[layer_idx] = act_layerwise_bits[layer_idx]
            edge_df['bit_activations'] = edge_act_bits_no_compression

        # Update weight bits - cloud
        num_cloud_layers = len(self.net_df) - num_edge_layers
        cloud_layer_bits = [self.CLOUD_BITS] * num_cloud_layers
        cloud_df['bit_weights'] = cloud_layer_bits
        cloud_df['bit_activations'] = cloud_layer_bits
        return edge_df, cloud_df

    def gen_stats_all_on_cloud(self):
        self.logger_auto.info('#################')
        self.logger_auto.info('All on Cloud')
        self.logger_auto.info('#################')
        layer_idx = -1
        edge_df = self.net_df[0:layer_idx + 1].copy()
        cloud_df = self.net_df[layer_idx + 1:].copy()
        cloud_df.reset_index(inplace=True)
        cloud_layer_bits = [self.CLOUD_BITS] * (len(self.net_df) - (layer_idx + 1))
        cloud_df['bit_weights'] = cloud_layer_bits
        cloud_df['bit_activations'] = cloud_layer_bits

        transmitted_act_bit = 0
        total_mse = self.get_mse(edge_df, cloud_df)
        name = '{}_{}_{}_False_False'.format(self.CLOUD_BITS, layer_idx, self.CLOUD_BITS)
        self.logger_auto.info('--- {} : mse:{} ---'.format(name, total_mse))
        if self.get_latency_stats:
            runtime_sec, edge_sec, transmission_sec, cloud_sec \
                = self.model_latency.get_model_latency(edge_df, cloud_df, layer_idx, name)
            self.logger_auto.debug('total_sec: {} edge_sec: {} tr_sec: {}'.format(
                runtime_sec,edge_sec,transmission_sec))
        else:
            runtime_sec = None
            edge_sec = None
            transmission_sec = None
            cloud_sec = None

        self.stats = self.stats.append({'name': name, 'mse': total_mse,
                                        'wgt_bits': self.CLOUD_BITS,
                                        'split_idx': layer_idx, 'model_size_KB': 0, 'latency':runtime_sec,
                                        'edge_sec': edge_sec, 'tr_sec': transmission_sec, 'cloud_sec': cloud_sec,
                                        'act_bits': self.CLOUD_BITS, 'tr_bits': 8,
                                        'no_act_compr': False,
                                        'act_opt': False
                                        },
                                       ignore_index=True)
        # self.logger_auto.debug(self.stats)
        self.dump_bit_configurations(name, edge_df, cloud_df)

    def gen_stats_all_on_edge(self, bitlist=(2,4,6,8) ):
        self.logger_auto.info('#################')
        self.logger_auto.info('All on Edge '.format(bitlist))
        self.logger_auto.info('#################')
        num_layers = self.weight_memory_solver.orig_dnn.num_layers
        layer_idx = num_layers - 1

        cloud_df = self.net_df[num_layers:].copy()
        cloud_df.reset_index(inplace=True)

        for avg_bits in bitlist:
            edge_df = self.net_df[0:num_layers].copy()
            edge_df.reset_index(inplace=True)
            edge_layer_bits = [avg_bits] * num_layers
            edge_df['bit_weights'] = edge_layer_bits
            edge_df['bit_activations'] = edge_layer_bits

            total_wgt_vol = edge_df['wgt_vol'].sum()*avg_bits
            total_act_vol = edge_df['ifm_vol'].max()*avg_bits
            model_size_KB = (total_act_vol+total_wgt_vol)/1024/8

            total_mse = self.get_mse(edge_df,cloud_df)
            name = '{}_{}_{}_False_False'.format(avg_bits, layer_idx, avg_bits)
            self.logger_auto.info('--- {} : mse:{} ---'.format(name, total_mse))
            if self.get_latency_stats:
                runtime_sec, edge_sec, transmission_sec, cloud_sec \
                    = self.model_latency.get_model_latency(edge_df, cloud_df, layer_idx, name)
                self.logger_auto.debug('total_sec: {} edge_sec: {} tr_sec: {}'.format(runtime_sec,edge_sec,transmission_sec))
            else:
                runtime_sec = None
                edge_sec = None
                transmission_sec = None
                cloud_sec = None

            self.stats = self.stats.append({'name': name, 'mse': total_mse,
                                            'wgt_bits': avg_bits,
                                            'split_idx': layer_idx, 'model_size_KB': model_size_KB, 'latency':runtime_sec,
                                            'edge_sec': edge_sec, 'tr_sec': transmission_sec, 'cloud_sec': cloud_sec,
                                            'act_bits': avg_bits, 'tr_bits': 0,
                                            'no_act_compr': False,
                                            'act_opt': False
                                            },
                                           ignore_index=True)
            self.logger_auto.debug(self.stats)
            self.dump_bit_configurations(name, edge_df, cloud_df)

        return

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

def main(model_name,device_memory_KB, logidr=None, NUMLAYERS=None):
    input_file_name = root_dir + 'generated/hw_simulator/post_process2/' + model_name + '.csv'
    input_df = pd.read_csv(input_file_name, index_col=0)
    random.seed(190087)
    if NUMLAYERS is None:
        NUMLAYERS = len(input_df)

    input_df = input_df[0:NUMLAYERS]
    input_df['pred'] = input_df['pred'].apply(string_to_int_list)

    if logdir is None:
        print('No Logdir selected: Running Autosplit v1')
        # Fix type issues
        input_df = input_df.astype({'ifm_vol': int, 'ofm_vol': int,
                                    'wgt_vol': int, "mac": int,
                                    'bit_weights': int,
                                    'bit_activations': int,
                                    'rank': int})

    else:
        print('Logdir selected: {} Running Autosplit v2'.format(logdir))
        input_df = input_df.astype({'ifm_vol': int, 'ofm_vol': int,
                                    'wgt_vol': int, "mac": int,
                                    'bit_weights': int,
                                    'bit_activations': int,
                                    'rank': int})

    dnn_weights_d = AttrDict()
    dnn_weights_d['data'] = OrderedDict()

    layer_names_df = input_df['name']
    weight_vol_df = input_df['wgt_vol']
    total_wgt_vol = weight_vol_df.sum()
    is_classification_model = False
    if logdir is None:
        dnn_weights_d.layers_df = pd.DataFrame(index=layer_names_df, columns=['vol','min','max'])
        wgt_data_df = None
        wgt_stats_df = None
    else:
        dnn_weights_d.layers_df = pd.DataFrame(index=layer_names_df,
                                               columns=['name', 'vol', 'min', 'max', 'orig_layer_idx',
                                                        'scale', 'zero_point', 'clipping'])
        wgt_data_df = pd.read_pickle(logidr + '/weights_data.pkl')
        wgt_stats_df = pd.read_pickle(logidr + '/weights_stats.pkl')
        if 'module.' in wgt_data_df['name'][0]:
            is_classification_model = True
        


    for layer_name, wgt_vol in zip(layer_names_df, weight_vol_df):
        if logdir is None:
            dnn_weights_d['data'][layer_name] = np.random.uniform(-1, 1, wgt_vol)
            dnn_weights_d.layers_df.loc[layer_name] = {'vol': wgt_vol, 'min': -1, 'max': 1}
        else:

            # Classification models have module in their name.
            # Detection models dont.
            if is_classification_model:
                distiller_name = 'module.' + layer_name + '.wrapped_module'
            else:
                distiller_name = layer_name + '.wrapped_module'
            wgt_data_np = wgt_data_df[wgt_data_df['name'] == distiller_name]['data'].to_numpy()[0]
            assert len(wgt_data_np) == wgt_vol
            dnn_weights_d['data'][layer_name] = wgt_data_np
            wgt_min = wgt_data_np.min()
            wgt_max = wgt_data_np.max()
            wgt_clipping = wgt_stats_df['clipping'][0]
            # wgt_scale = wgt_stats_df['scale'][0]
            # wgt_zero_point = wgt_stats_df['zero_point'][0]
            wgt_scale = None
            wgt_zero_point = None

            dnn_weights_d.layers_df.loc[layer_name] = {'vol': wgt_vol, 'min': wgt_min, 'max': wgt_max,
                                                       # 'avg_min': min_act, 'avg_max': max_act,
                                                       'scale': wgt_scale, 'zero_point': wgt_zero_point,
                                                       'clipping': wgt_clipping
                                                       }

    # activations
    dnn_act_d = AttrDict()
    dnn_act_d['data'] = OrderedDict()
    act_vol_df = input_df['ofm_vol']
    total_act_vol = act_vol_df.sum()
    if logdir is None:
        dnn_act_d.layers_df = pd.DataFrame(index=layer_names_df, columns=['vol','min','max',
                                                                          'avg_min','avg_max',
                                                                          'mean','std','b'])
    else:
        dnn_act_d.layers_df = pd.DataFrame(index=layer_names_df, columns=['vol', 'min', 'max',
                                                                          'scale', 'zero_point',
                                                                          'clipping'])

    input_image_shape= re.split('[()]', input_df['ifm'][0])[1:-1]
    input_image_shape= input_image_shape[0].split(', ')
    input_image_vol=  np.prod([int(x) for x in input_image_shape])
    input_image_vol_bits = input_image_vol*8

    for layer_name, act_vol in zip(layer_names_df, act_vol_df):
        if logdir is None:
            # Assuming relu6
            max_act = random.uniform(0.7, 6)
            dnn_act_d['data'][layer_name] = np.random.uniform(0, max_act, act_vol)
            dnn_act_d.layers_df.loc[layer_name] = {'vol': act_vol, 'min': 0.7, 'max': 6,
                                                   'avg_min': 0, 'avg_max': 6,
                                                   'mean': 2, 'std': 3, 'b': 0.233
                                                   }
        else:
            # Classification models have module in their name.
            # Detection models dont.
            if is_classification_model:
                distiller_name = 'module.' + layer_name
            else:
                distiller_name = layer_name

            out_act_df = pd.read_pickle(logidr + '/out_act/' + distiller_name + '/act_data.pkl')
            dnn_act_d['data'][layer_name] = out_act_df['data'][0]
            act_stats = pd.read_pickle(logidr + '/out_act/' + distiller_name + '/act_stats.pkl')
            min_act = act_stats['min'][0]
            max_act = act_stats['max'][0]
            clipping = act_stats['clipping'][0]
            # scale = act_stats['scale'][0]
            # zero_point = act_stats['zero_point'][0]
            scale = None
            zero_point = None
            act_vol = act_stats['vol'][0]
            dnn_act_d.layers_df.loc[layer_name] = {'vol': act_vol, 'min': min_act, 'max': max_act,
                                                   'avg_min': min_act, 'avg_max': max_act,
                                                   'scale': scale, 'zero_point': zero_point,
                                                   'clipping': clipping
                                                   }

    # run Memory Constraint Solver
    bitlist = np.array([1, 2, 4, 5, 6, 7, 8])
    bitlist_cloud = np.array([16, 32])

    print(datetime.datetime.now())
    # DEBUG_SPLIT_SPP = [9, 11, 13, 15, 17]
    DEBUG_SPLIT_TINY = [1, 9, 11]
    DEBUG_SPLIT=None
    AutoSplit(model_name, input_df, dnn_act_d, dnn_weights_d, bitlist, bitlist_cloud,
              total_memory_size_KB=device_memory_KB,
              lambda_step_mul_const=2, lambda_step_div_const=15, image_size=input_image_vol_bits,
              is_auto_split=True, DEBUG_SPLIT=DEBUG_SPLIT)
    print(datetime.datetime.now())

if __name__ == '__main__':

    print(datetime.datetime.now())
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', type=lambda s: s.lower(),
                        help='add graph information to model stats dataframe')
    # logs/resnet50_stats
    parser.add_argument('--logdir', metavar='LOGDIR', default=None, type=lambda s: s.lower(),
                        help='logdir to load weights_stats.pkl, weights_data.pkl, '
                             'out_act/act_data.pkl  out_act/act_stats.pkl')
    args = parser.parse_args()
    print('Running model: {}'.format(args.arch))
    sel_model_name=args.arch
    logdir = args.logdir

    device_memory_KB_per_model = {
        'CONV2D': 2**10,
        'yolov3-416': 2**17, 'yolov3-512': 2**18, 'yolov3-608': 2**19,
        'yolov3-spp-416': 2**17, 'yolov3-spp-512': 2**18,'yolov3-spp-608': 2**19,
        'yolov3-tiny-416': 2 ** 15, 'yolov3-tiny-512': 2 ** 15, 'yolov3-tiny-608': 2 ** 15,
        'resnet18':2**10 , 'resnet_fpn_graph': 2**12,'resnet50': 2**12, 'mobilenet_v2': 2**10,
        'xception': 2 ** 14,
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
    main(model_name,device_memory_KB, logdir , NUMLAYERS = num_layers)

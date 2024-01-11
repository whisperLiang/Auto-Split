
from attrdict import AttrDict
import random
import math
import numpy as np
import enum
import os
import datetime
from pathlib import Path
import pandas as pd
import copy
from tools.hw_simulator.schedule.net import Net
from tools.bit_search.linear_quantization import Quantize
# from bit_constraint_solver import Status
from collections import OrderedDict
import itertools
LARGENUMBER=999999
import pandas as pd
import logging, sys

class Status(enum.Enum):
    lambda_min_set = 0
    lambda_max_set = 1
    both_lambda_set = 2
    soln_reached_check_lambda_max = 3
    soln_reached_check_lambda = 4
    memory_too_small = 5
    max_loop_reached=6


class MemorySolver():

    def __init__(self, DNN_D, avg_bit_constraint, bitlist = np.array([1, 2, 4, 8]), bitlist_cloud = np.array([16,32]),
                 lambda_step_mul_const=2, lambda_step_div_const=15,logger=None):


        if logger is None:
            logging.basicConfig(stream=sys.stderr, level=logging.INFO)
            self.logger = logging.getLogger('logs')
        else:
            self.logger = logger

        # Constants
        self.lambda_step = lambda_step_div_const
        self.lambda_step_div_const = lambda_step_div_const
        self.lambda_step_mul_const = lambda_step_mul_const

        # ---
        self.quantize = Quantize()
        self.initialize_dnn_state(bitlist, bitlist_cloud, DNN_D, avg_bit_constraint)

        self.logger.debug('Average bit constraint: {} mem_bits or {} KB'.format(self.dnn_state_D.avg_bit_constraint,
                                                                     self.dnn_state_D.total_mem_constraint_KB
                                                                     ))
        self.set_mse(DNN_D)
        self.var_lambda_init = self.get_lambda_init()

    def reset_dnn_state(self, selected_num_layers, available_mem, isKB=True):
        self.dnn_state_D = None
        self.dnn_state_D = AttrDict()
        self.dnn_state_D['vol'] = OrderedDict()
        self.dnn_state_D['var_lambda_min'] = None
        self.dnn_state_D['var_lambda_max'] = None
        self.dnn_state_D['var_lambda'] = None
        self.dnn_state_D['total_bits_lambda_min'] = None
        self.dnn_state_D['total_bits_lambda_max'] = None

        selected_dnn = AttrDict()
        selected_dnn['num_layers'] = selected_num_layers
        selected_dnn.layers_df = copy.deepcopy(self.orig_dnn.layers_df.iloc[0:selected_num_layers])
        total_vol = selected_dnn.layers_df['vol'].sum()
        self.dnn_state_D['total_vol'] = total_vol
        self.dnn_state_D.selected_dnn = selected_dnn

        if isKB:
            self.dnn_state_D.avg_bit_constraint = available_mem* 1024 * 8 /total_vol
        else:
            self.dnn_state_D.avg_bit_constraint = available_mem
        self.dnn_state_D['selected_num_layers'] =  selected_num_layers

        self.dnn_state_D['total_mem_constraint_KB'] = self.dnn_state_D.avg_bit_constraint * total_vol / 8 / 1024

        # ----
        # TODO select only num_layers
        layer_names = list(selected_dnn.layers_df.index)
        vol = list(selected_dnn.layers_df['vol'])

        for layer_name, vol in zip(layer_names, vol):
            membits = np.array(self.bitlist)*vol/total_vol
            self.dnn_state_D['vol'][layer_name]  = {'mem_bits': membits,
                                                   'bits': self.bitlist, 'lambda_bit': None,
                                                    'lambda_min_bit': None,
                                                   'lambda_max_bit': None, 'vol': vol}

        # self.logger.debug('## RESET avg_bit_constraint: {}\t selected_num_layers: {}'.format(self.dnn_state_D.avg_bit_constraint,
        #                                                                       selected_num_layers))


    def initialize_dnn_state(self, bitlist, bitlist_cloud, DNN_D, avg_bit_constraint):
        self.bitlist = bitlist
        self.bitlist_cloud = bitlist_cloud

        self.dnn_state_D = AttrDict()
        self.dnn_state_D['vol'] = OrderedDict()
        self.dnn_state_D['var_lambda_min'] = None
        self.dnn_state_D['var_lambda_max'] = None
        self.dnn_state_D['var_lambda'] = None
        self.dnn_state_D['total_bits_lambda_min'] = None
        self.dnn_state_D['total_bits_lambda_max'] = None

        num_layers = len(DNN_D.layers_df)
        
        self.dnn_state_D['avg_bit_constraint'] = avg_bit_constraint
        self.dnn_state_D['selected_num_layers'] =  num_layers
        total_vol = DNN_D.layers_df['vol'].sum()
        self.dnn_state_D['total_vol'] = total_vol
        self.dnn_state_D['total_mem_constraint_KB'] = self.dnn_state_D.avg_bit_constraint * total_vol / 8 / 1024

        self.orig_dnn = AttrDict()
        self.orig_dnn['num_layers'] = num_layers
        self.orig_dnn['total_vol'] = total_vol
        self.orig_dnn.layers_df = copy.deepcopy(DNN_D.layers_df)

        # ----
        layer_names = list(DNN_D.layers_df.index)
        vol = list(DNN_D.layers_df['vol'])

        for layer_name, vol in zip(layer_names, vol):
            membits = np.array(bitlist)*vol/total_vol
            self.dnn_state_D['vol'][layer_name]  = {'mem_bits': membits,
                                                   'bits': bitlist, 'lambda_bit': None, 'lambda_min_bit': None,
                                                   'lambda_max_bit': None, 'vol': vol}



    def set_mse(self, DNN_D):
        total_vol = self.orig_dnn.total_vol
        bitlist = np.concatenate([self.bitlist, self.bitlist_cloud])
        cols = [str(x) for x in bitlist]
        layer_names = [key for key, value in DNN_D.data.items()]
        self.dnn_mse_df = pd.DataFrame(index=layer_names, columns=cols)

        # get sat_min, sat_max
        # For weights -1,1; for act: 0,6 for now

        for wgt, value in DNN_D.data.items():
            orig_weight_vector = value
            sat_min = DNN_D.layers_df.loc[wgt]['min']
            sat_max = DNN_D.layers_df.loc[wgt]['max']

            if 'scale' in DNN_D.layers_df:
                scale = DNN_D.layers_df.loc[wgt]['scale']
                zero_point = DNN_D.layers_df.loc[wgt]['zero_point']
                clipping = DNN_D.layers_df.loc[wgt]['clipping']
            else:
                scale = None
                zero_point = None
                clipping = None
            mse_per_bit = self.get_mse_per_param(bitlist, orig_weight_vector,
                                                 sat_min, sat_max, scale, zero_point, clipping) / total_vol
            self.dnn_mse_df.loc[wgt] = list(mse_per_bit)

        # print(self.dnn_mse_df)
        # print('---------')

    def get_lambda_init(self):
        min_bit = self.bitlist[0]
        max_bit = self.bitlist[-1]
        sum_min_bit_mse = self.dnn_mse_df[str(min_bit)].sum()
        sum_max_bit_mse = self.dnn_mse_df[str(max_bit)].sum()
        sum_mse_diff = abs(sum_min_bit_mse - sum_max_bit_mse)
        lambda_init = (1.0/self.dnn_state_D.total_vol)*sum_mse_diff
        # lambda_init = sum_mse
        # self.logger.debug('lambda_init: {}'.format(lambda_init))
        assert lambda_init >= 0, 'var_lambda_init is negative'
        assert sum_mse_diff != 0, 'sum_mse_diff == 0'
        return lambda_init

    def mem_constraint_solver(self):
        self.dnn_state_D.var_lambda = self.var_lambda_init
        status = None # 'min', 'max', 'soln'
        MAXLOOPS= 500
        num_loop_idx=0
        while(1):
            num_loop_idx +=1
            if num_loop_idx >= MAXLOOPS:
                status = Status.max_loop_reached
                break
            if status == Status.both_lambda_set:
                # self.logger.debug(' --- iterate over lambda range --- ')
                break
            elif status == Status.soln_reached_check_lambda_max:
                # self.logger.debug('solution reached -check lambda_max_bit')
                break
            elif status == Status.soln_reached_check_lambda:
                # self.logger.debug('solution reached -check lambda')
                break
            elif status == Status.memory_too_small:
                self.logger.warn('solution reached -check lambda')
                break
            else:
                # get total bits and set lambda_bit
                total_bits_lambda = self.set_lambda_bits(self.dnn_state_D.var_lambda)
                if total_bits_lambda is None:
                    status = Status.memory_too_small
                    break
                total_bits_lambda = round(total_bits_lambda,1)
                # self.logger.debug(total_bits_lambda, self.dnn_state_D.avg_bit_constraint)
                if total_bits_lambda <= self.dnn_state_D.avg_bit_constraint:
                    # use eq(11) to get lambda_one
                    var_lambda_new = self.get_lambda_max()
                    assert var_lambda_new >= 0, 'var_lambda_new is negative !!!'
                    if var_lambda_new==0:
                        # raise ValueError('var_lambda_new ==0 => entire model with max available bits will fit')
                        status = Status.soln_reached_check_lambda
                        continue
                    total_bits_lambda_new = self.set_lambda_bits(var_lambda_new)
                    if total_bits_lambda_new is None:
                        status = Status.memory_too_small
                        break
                    status = self.set_lambda_min_max(total_bits_lambda_new, var_lambda_new)
                if total_bits_lambda > self.dnn_state_D.avg_bit_constraint:
                    var_lambda_new = self.get_lambda_min()
                    assert var_lambda_new > 0, 'var_lambda_new is negative !!!'
                    total_bits_lambda_new = self.set_lambda_bits(var_lambda_new)
                    if total_bits_lambda_new is None:
                        status = Status.memory_too_small
                        break
                    status = self.set_lambda_min_max(total_bits_lambda_new, var_lambda_new)

                # Update next lambda
                if status == Status.lambda_min_set:
                    # self.logger.debug('lambda_min set: {}, total mem_bits: {}'.format(self.dnn_state_D.var_lambda_min, self.dnn_state_D.total_bits_lambda_min))
                    self.lambda_step = abs(self.dnn_state_D.var_lambda - self.dnn_state_D.var_lambda_min)*self.lambda_step_mul_const
                    # self.lambda_step = abs(self.dnn_state_D.var_lambda_min) * self.lambda_step_mul_const
                    self.dnn_state_D.var_lambda = self.dnn_state_D.var_lambda + self.lambda_step

                if status == Status.lambda_max_set:
                    # self.logger.debug('lambda_max set: {}, total mem_bits: {}'.format(self.dnn_state_D.var_lambda_max, self.dnn_state_D.total_bits_lambda_max))
                    self.lambda_step = abs(self.dnn_state_D.var_lambda - self.dnn_state_D.var_lambda_max)*self.lambda_step_mul_const
                    if self.lambda_step >= self.dnn_state_D.var_lambda:
                        # raise ValueError('var_lambda: {}, lambda_max: {}, step: {}'.format(
                        #     self.dnn_state_D.var_lambda, self.dnn_state_D.var_lambda_max, self.lambda_step))
                        self.lambda_step = abs(self.dnn_state_D.var_lambda/self.lambda_step_mul_const)
                        # self.lambda_step = abs(self.dnn_state_D.var_lambda_max) * self.lambda_step_mul_const
                    self.dnn_state_D.var_lambda = self.dnn_state_D.var_lambda - self.lambda_step


                # if status == Status.both_lambda_set:
                    # self.logger.debug('lambda_min set: {}, total mem_bits: {} ||| lambda_max set: {}, total mem_bits: {}'.format(
                    #     self.dnn_state_D.var_lambda_min, self.dnn_state_D.total_bits_lambda_min,
                    #     self.dnn_state_D.var_lambda_max, self.dnn_state_D.total_bits_lambda_max))

                if self.dnn_state_D.var_lambda <= 0:
                    raise AssertionError('var_lambda is negative!!!')

        if status == Status.both_lambda_set:
            lambda_min_diff_bits = self.dnn_state_D.total_bits_lambda_min - self.dnn_state_D.avg_bit_constraint
            lambda_max_diff_bits = self.dnn_state_D.avg_bit_constraint - self.dnn_state_D.total_bits_lambda_max

            if lambda_max_diff_bits >= lambda_min_diff_bits:
                return self.iterate_bits_range_min()
            else:
                return self.iterate_bits_range_max()

        elif status == Status.soln_reached_check_lambda_max:
            # self.logger.debug('solution reached -check lambda_max_bit')

            return self.print_solution(selected_lambda='lambda_max_bit')

        elif status == Status.soln_reached_check_lambda:
            # self.logger.debug('solution reached -check lambda_bit')
            return self.print_solution(selected_lambda='lambda_bit')

        elif status == Status.memory_too_small:
            self.logger.warn('solution reached -check lambda')
            return self.print_solution(selected_lambda=None, status=status)
        elif status == Status.max_loop_reached:
            self.logger.warn('MAX loop reached. No Solution')
            return self.print_solution(selected_lambda=None, status=status)

    def iterate_bits_range_min(self):
        # Start from lambda_max and obtain successive lambda by solvinf 10(c) until R(lambda) < Rc
        self.dnn_state_D.var_lambda = self.dnn_state_D.var_lambda_min
        total_bits_lambda = self.set_lambda_bits(self.dnn_state_D.var_lambda)
        # self.logger.debug('var_lambda: {} total_bits_lambda: {}'.format(self.dnn_state_D.var_lambda, total_bits_lambda))
        # update lambda step for the next iterations
        self.lambda_step = (self.dnn_state_D.var_lambda_max - self.dnn_state_D.var_lambda_min) /self.lambda_step_div_const
        old_var_lambda = self.dnn_state_D.var_lambda
        old_total_bits_lambda = total_bits_lambda

        while total_bits_lambda >= self.dnn_state_D.avg_bit_constraint:
            # use eq(11) to get lambda_one
            var_lambda_new = self.get_lambda_min_range()
            assert var_lambda_new <= self.dnn_state_D.var_lambda_max, ' reached var_lambda_max !!! {}'.format(var_lambda_new)
            total_bits_lambda_new = self.set_lambda_bits(var_lambda_new)

            # Update next lambda
            old_var_lambda = self.dnn_state_D.var_lambda
            old_total_bits_lambda = total_bits_lambda
            self.dnn_state_D.var_lambda = self.dnn_state_D.var_lambda + self.lambda_step
            # if self.dnn_state_D.var_lambda > self.dnn_state_D.var_lambda_max:
            #     raise AssertionError('var_lambda is too big!!!')
            total_bits_lambda = self.set_lambda_bits(self.dnn_state_D.var_lambda)
            # self.logger.debug('var_lambda: {} total_bits_lambda: {}'.format(self.dnn_state_D.var_lambda, total_bits_lambda))

        # solution is at self.dnn_state_D.var_lambda
        if old_total_bits_lambda == self.dnn_state_D.avg_bit_constraint:
            self.dnn_state_D.var_lambda = old_var_lambda
            total_bits_lambda = self.set_lambda_bits(self.dnn_state_D.var_lambda)
            # self.logger.debug('soln: var_lambda: {} total_bits_lambda: {}'.format(self.dnn_state_D.var_lambda, total_bits_lambda))

        return self.print_solution()

    def iterate_bits_range_max(self):
        # Start from lambda_max and obtain successive lambda by solvinf 10(c) until R(lambda) < Rc
        self.dnn_state_D.var_lambda = self.dnn_state_D.var_lambda_max
        total_bits_lambda = self.set_lambda_bits(self.dnn_state_D.var_lambda)
        # self.logger.debug('var_lambda: -- total_bits_lambda: -- ')
        # self.logger.debug('var_lambda: {} total_bits_lambda: {}'.format(self.dnn_state_D.var_lambda, total_bits_lambda))
        # update lambda step for the next iterations
        self.lambda_step = (self.dnn_state_D.var_lambda_max - self.dnn_state_D.var_lambda_min) / self.lambda_step_div_const
        old_var_lambda = self.dnn_state_D.var_lambda
        while total_bits_lambda <= self.dnn_state_D.avg_bit_constraint:
            # use eq(11) to get lambda_one
            var_lambda_new = self.get_lambda_max_range()
            assert var_lambda_new >= self.dnn_state_D.var_lambda_min, 'var_lambda_new is negative'
            total_bits_lambda_new = self.set_lambda_bits(var_lambda_new)
            # Update next lambda
            old_var_lambda = self.dnn_state_D.var_lambda
            self.dnn_state_D.var_lambda = self.dnn_state_D.var_lambda - self.lambda_step
            total_bits_lambda = self.set_lambda_bits(self.dnn_state_D.var_lambda)
            # self.logger.debug('var_lambda: {} total_bits_lambda: {}'.format(self.dnn_state_D.var_lambda, total_bits_lambda))

        # solution is at self.dnn_state_D.var_lambda
        self.dnn_state_D.var_lambda = old_var_lambda
        total_bits_lambda = self.set_lambda_bits(self.dnn_state_D.var_lambda)
        # self.logger.debug('soln: var_lambda: {} total_bits_lambda: {}'.format(self.dnn_state_D.var_lambda, total_bits_lambda))
        return self.print_solution()

    #-----------------------------
    # Auxiliary functions
    #-----------------------------
    def get_mse(self, name_bits_df):

        total_mse = 0
        for index, row in name_bits_df.iterrows():
            name, bits = row
            # bit_idx = np.where(self.bitlist == bits)[0][0]
            total_mse += self.dnn_mse_df.loc[name][str(bits)]

        return total_mse

    def print_solution(self, selected_lambda=None, status=None):

        if status == Status.memory_too_small or status == Status.max_loop_reached:
            return None

        if selected_lambda is None or selected_lambda == 'lambda_bit':
            selected_lambda = 'lambda_bit'
        elif selected_lambda == 'lambda_min_bit':
            self.dnn_state_D.var_lambda = self.dnn_state_D.var_lambda_min
        elif selected_lambda == 'lambda_max_bit':
            self.dnn_state_D.var_lambda = self.dnn_state_D.var_lambda_max
        else:
            raise ValueError ('selected_lambda value is incorrect !!!')

        # print lambda bit distribution
        total_mem_bits = 0
        total_bits = 0
        total_mse = 0
        total_mse_minbit = 0
        total_mse_maxbit = 0
        layerwise_selected_bits = OrderedDict()
        layerwise_min_bits = OrderedDict()
        layerwise_max_bits = OrderedDict()
        for wgt, value in self.dnn_state_D.vol.items():
            selected_mem_bit = value[selected_lambda]
            selected_bit_idx = np.where(value['mem_bits'] == selected_mem_bit)[0][0]
            selected_bit = value['bits'][selected_bit_idx]
            layerwise_selected_bits[wgt] = selected_bit
            layerwise_min_bits[wgt] = value['bits'][0]
            layerwise_max_bits[wgt] = value['bits'][-1]
            vol_percent = round(100* value['vol']/self.dnn_state_D.total_vol,2)
            self.logger.debug('{} mem_bits: {}, bits: {}  vol%: {}'.format(wgt, selected_mem_bit, selected_bit, vol_percent ))
            total_mem_bits += selected_mem_bit
            total_bits += selected_bit*value['vol']
            total_mse += self.dnn_mse_df.loc[wgt][str(self.bitlist[selected_bit_idx])]
            total_mse_minbit += self.dnn_mse_df.loc[wgt][str(self.bitlist[0])]
            total_mse_maxbit += self.dnn_mse_df.loc[wgt][str(self.bitlist[-1])]

        self.logger.debug('memory_KB: {} total_mem_bits: {} total_mse: {}, total_mse_minbit: {}, '
              'total_mse_maxbit: {}'.format(total_bits/1024, total_mem_bits, total_mse,
                                            total_mse_minbit, total_mse_maxbit))
        min_bit = self.bitlist[0]
        max_bit = self.bitlist[-1]
        table_entry = [{'memory_KB': total_bits/1024, 'mem_bits': total_mem_bits, 'mse': total_mse,
                        'selected_num_layers': self.dnn_state_D.selected_num_layers,
                        'layerwise_selected_bits': layerwise_selected_bits},
                       {'memory_KB': min_bit*self.dnn_state_D.total_vol/1024, 'mem_bits': min_bit,
                        'mse': total_mse_minbit, 'selected_num_layers': self.dnn_state_D.selected_num_layers,
                        'layerwise_selected_bits': layerwise_min_bits},
                       {'memory_KB': max_bit*self.dnn_state_D.total_vol/1024, 'mem_bits': max_bit,
                        'mse': total_mse_maxbit, 'selected_num_layers': self.dnn_state_D.selected_num_layers,
                        'layerwise_selected_bits': layerwise_max_bits}]

        table = pd.DataFrame(table_entry, columns=['memory_KB', 'mem_bits', 'mse','selected_num_layers',
                                                   'layerwise_selected_bits'])
        return table

    def get_mse_per_param(self, bit, orig_weight_vector, sat_min, sat_max,
                          scale=None, zero_point=None, clipping=None):
        # quantize the selected weight to the given bits
        # and calculate mse loss.
        if scale is None or zero_point or None or clipping is None:
            weight_vector_q = self.quantize.quantize(orig_weight_vector, bit, sat_min, sat_max)
        else:
            weight_vector_q = self.quantize.quantize(orig_weight_vector, bit,
                                                     sat_min, sat_max,
                                                     scale, zero_point, clipping)

        mse_loss = np.sqrt(np.sum((orig_weight_vector.reshape(1, -1) - weight_vector_q) ** 2, axis=1))
        # print('bits:{} mse_loss: {}'.format(bit, mse_loss))
        return mse_loss

    def set_lambda_bits(self, var_lambda):
        total_min_mem_bits = 0
        num_lowest_bits = 0
        wgt = None
        # sorted_min_mem_bits = None
        # sorted_bits = None
        for wgt, value in self.dnn_state_D['vol'].items():
            mem_bits = self.dnn_state_D.vol[wgt]['mem_bits']
            bits = self.dnn_state_D.vol[wgt]['bits']
            b = mem_bits*var_lambda
            len_b = len(b)
            loss = list(self.dnn_mse_df.loc[wgt][:len_b]) + b

            sorted_loss, sorted_min_mem_bits, sorted_bits = \
                (list(t) for t in zip(*sorted(zip(loss, mem_bits, bits),
                                              key=lambda x: (x[0], x[1]))))
            self.dnn_state_D.vol[wgt]['lambda_bit'] = sorted_min_mem_bits[0]
            # print('mem_bits: {}, bits: {}'.format(sorted_min_mem_bits[0], sorted_bits[0] ))
            total_min_mem_bits += sorted_min_mem_bits[0]
            if sorted_bits[0] == bits[0]:
                num_lowest_bits += 1

        if num_lowest_bits == len(self.dnn_state_D.vol):
            # print('{}: lambda_bit: mem_bit:{} bit:{}'.format(wgt, sorted_min_mem_bits[0], sorted_bits[0]))
            self.logger.warn('memory is too small -- everything is lowest bit -- ')
            total_min_mem_bits=None

        return total_min_mem_bits

    def get_lambda_max(self):
        var_lambda_new = 0
        for wgt, value in self.dnn_state_D.vol.items():
            prev_lambda_bit = value['lambda_bit']
            prev_lambda_bit_idx = np.where(value['mem_bits'] == prev_lambda_bit)[0][0]
            prev_lambda_mse = self.dnn_mse_df.loc[wgt][str(self.bitlist[prev_lambda_bit_idx])]
            mem_bits = value['mem_bits']
            if prev_lambda_bit_idx +1 == len(mem_bits) :
                continue
            for mem_bit in mem_bits[prev_lambda_bit_idx+1:]:
                bit_idx = np.where(value['mem_bits'] == mem_bit)[0][0]
                mse = self.dnn_mse_df.loc[wgt][str(self.bitlist[bit_idx])]
                potential_lambda = (prev_lambda_mse - mse)/(mem_bit - prev_lambda_bit)
                # select global maximum
                if var_lambda_new <= potential_lambda:
                    var_lambda_new = potential_lambda

        return var_lambda_new

    def get_lambda_min(self):
        var_lambda_new = LARGENUMBER
        for wgt, value in self.dnn_state_D.vol.items():
            prev_lambda_bit = value['lambda_bit']
            prev_lambda_bit_idx = np.where(value['mem_bits'] == prev_lambda_bit)[0][0]
            prev_lambda_mse = self.dnn_mse_df.loc[wgt][str(self.bitlist[prev_lambda_bit_idx])]
            mem_bits = value['mem_bits']
            if prev_lambda_bit_idx == 0:
                continue
            for mem_bit in mem_bits[0:prev_lambda_bit_idx]:
                bit_idx = np.where(value['mem_bits'] == mem_bit)[0][0]
                mse = self.dnn_mse_df.loc[wgt][str(self.bitlist[bit_idx])]
                potential_lambda = (prev_lambda_mse - mse)/(mem_bit - prev_lambda_bit)
                # Select global minimum
                if var_lambda_new >= potential_lambda:
                    var_lambda_new = potential_lambda

        return var_lambda_new

    def get_lambda_min_range(self):
        var_lambda_new = self.dnn_state_D.var_lambda_max
        for wgt, value in self.dnn_state_D.vol.items():
            lambda_max_bit = value['lambda_max_bit']
            lambda_max_bit_idx = np.where(value['mem_bits'] == lambda_max_bit)[0][0]

            # prev_lambda
            prev_lambda_bit = value['lambda_bit']
            prev_lambda_bit_idx = np.where(value['mem_bits'] == prev_lambda_bit)[0][0]
            prev_lambda_mse = self.dnn_mse_df.loc[wgt][str(self.bitlist[prev_lambda_bit_idx])]

            mem_bits = value['mem_bits']

            if lambda_max_bit_idx >= prev_lambda_bit_idx and lambda_max_bit_idx >= prev_lambda_bit_idx:
                continue

            for bit in mem_bits[lambda_max_bit_idx:prev_lambda_bit_idx]:
                bit_idx = np.where(value['mem_bits'] == bit)[0][0]
                mse = self.dnn_mse_df.loc[wgt][str(self.bitlist[bit_idx])]
                potential_lambda = (prev_lambda_mse - mse) / (bit - prev_lambda_bit)
                # select global maximum
                if var_lambda_new >= potential_lambda:
                    var_lambda_new = potential_lambda

        return var_lambda_new

    def get_lambda_max_range(self):
        var_lambda_new = self.dnn_state_D.var_lambda_min
        for wgt, value in self.dnn_state_D.vol.items():
            # lambda_min
            lambda_min_bit = value['lambda_min_bit']
            lambda_min_bit_idx = np.where(value['mem_bits'] == lambda_min_bit)[0][0]

            # lambda_max
            lambda_max_bit = value['lambda_max_bit']
            lambda_max_bit_idx = np.where(value['mem_bits'] == lambda_max_bit)[0][0]


            # prev_lambda
            prev_lambda_bit = value['lambda_bit']
            prev_lambda_bit_idx = np.where(value['mem_bits'] == prev_lambda_bit)[0][0]
            prev_lambda_mse = self.dnn_mse_df.loc[wgt][str(self.bitlist[prev_lambda_bit_idx])]

            mem_bits = value['mem_bits']

            if lambda_max_bit_idx >= lambda_min_bit_idx and prev_lambda_bit_idx >= lambda_min_bit_idx:
                continue

            for bit in mem_bits[prev_lambda_bit_idx+1:lambda_min_bit_idx+1]:
                bit_idx = np.where(value['mem_bits'] == bit)[0][0]
                mse = self.dnn_mse_df.loc[wgt][str(self.bitlist[bit_idx])]
                potential_lambda = (prev_lambda_mse - mse)/(bit - prev_lambda_bit)
                # select global maximum
                if var_lambda_new <= potential_lambda:
                    var_lambda_new = potential_lambda

        return var_lambda_new

    def set_lambda_min_max(self, total_bits_lambda_new, var_lambda_new):
        status = None
        if total_bits_lambda_new <= self.dnn_state_D.avg_bit_constraint:
            self.dnn_state_D.var_lambda_max = var_lambda_new
            self.dnn_state_D.total_bits_lambda_max = total_bits_lambda_new
            self.set_lambda_min_max_bit(is_lambda_min=False)
            status =  Status.lambda_max_set
        else:
            self.dnn_state_D.var_lambda_min = var_lambda_new
            self.dnn_state_D.total_bits_lambda_min = total_bits_lambda_new
            self.set_lambda_min_max_bit(is_lambda_min=True)
            status = Status.lambda_min_set

        # Both lambda_min and lambda_max is set
        if self.dnn_state_D.var_lambda_min is not None and self.dnn_state_D.var_lambda_max is not None:
            status = Status.both_lambda_set

        if total_bits_lambda_new == self.dnn_state_D.avg_bit_constraint:
            status = Status.soln_reached_check_lambda_max

        return status

    def set_lambda_min_max_bit(self, is_lambda_min):
        for wgt, value in self.dnn_state_D.vol.items():

            if is_lambda_min:
                self.dnn_state_D.vol[wgt]['lambda_min_bit'] = self.dnn_state_D.vol[wgt]['lambda_bit']
            else:
                self.dnn_state_D.vol[wgt]['lambda_max_bit'] = self.dnn_state_D.vol[wgt]['lambda_bit']






if __name__ == '__main__':
    print(datetime.datetime.now())
    model_names = ['CONV2D', 'resnet18']
    device = 'cloud'
    root_dir = os.getcwd() + '/'

    # -----
    # if device == 'edge':
    #     hardware_yaml = root_dir + "/tools/hw_simulator/schedule/params/hardware_config_edge.yaml"
    # elif device == 'cloud':
    #     hardware_yaml = root_dir + "/tools/hw_simulator/schedule/params/hardware_config_cloud.yaml"
    #
    # hardware_dict = None
    # verbose = False
    # for model_name in model_names[1:2]:
    #     input_file_name = root_dir + 'generated/hw_simulator/post_process/' + model_name + '.csv'
    #     df = pd.read_csv(input_file_name)
    #     random.seed(190087)
    #     # NUMLAYERS = len(df)
    #     NUMLAYERS = 5
    #     df = df[0:NUMLAYERS]
    #     dnn_weights_d = AttrDict()
    #     dnn_weights_d['vol'] = OrderedDict()
    #
    #     layer_names_df = df['name']
    #     weight_vol_df = df['wgt_vol']
    #     total_wgt_vol = weight_vol_df.sum()
    #     dnn_weights_d.layers_df = pd.DataFrame(index=layer_names_df, columns = ['vol'])
    #     dnn_weights_d['sat_min'] = -1
    #     dnn_weights_d['sat_max'] = 1
    #     for layer_name, wgt_vol in zip(layer_names_df,weight_vol_df):
    #         dnn_weights_d['vol'][layer_name] = np.random.uniform(-1,1,wgt_vol)
    #         dnn_weights_d.layers_df.loc[layer_name] = wgt_vol
    #
    #     # activations
    #     dnn_act_d = AttrDict()
    #     dnn_act_d['vol'] = OrderedDict()
    #     act_vol_df = df['ofm_vol']
    #     total_act_vol = act_vol_df.sum()
    #     dnn_act_d.layers_df = pd.DataFrame(index=layer_names_df, columns=['vol'])
    #     dnn_act_d['sat_min'] = 0
    #     dnn_act_d['sat_max'] = 6
    #     for layer_name, act_vol in zip(layer_names_df,act_vol_df):
    #         # Assuming relu6
    #         max_act = random.uniform(0.7,6)
    #         dnn_act_d['vol'][layer_name] = np.random.uniform(0,max_act,act_vol)
    #         dnn_act_d.layers_df.loc[layer_name] = act_vol
    #
    #     # run Memory Constraint Solver
    #     bitlist = np.array([1, 2, 4, 6, 8])
    #     bitlist_cloud = np.array([16,32])
    #     avg_bit_constraint = 4
    #     logging.basicConfig(stream=sys.stderr, level=logging.INFO)
    #     logger_weights = logging.getLogger('WGT')
    #     memory_solver = MemorySolver(dnn_weights_d, avg_bit_constraint, bitlist, bitlist_cloud,
    #                  lambda_step_mul_const=2,
    #                  lambda_step_div_const=15, logger=logger_weights)
    #     memory_solver.mem_constraint_solver()
    #     print(datetime.datetime.now())

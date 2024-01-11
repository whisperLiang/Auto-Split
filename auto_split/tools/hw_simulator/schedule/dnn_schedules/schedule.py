import yaml
from attrdict import AttrDict
import copy
from collections import OrderedDict
import csv
import os

class Schedule():
    # cwh_sched_0
    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):

        self.model_name = model_name
        self.net = net
        self.stats = OrderedDict()
        self.debug = verbose
        self.result_dir = result_dir
        self.BYTESIZE=8

        if hardware_yaml is not None:
            self.process_hardware_config(hardware_yaml)
        else:
            self.process_hardware_config_dict(hardware_dict)

        if hardware_dict is None and hardware_yaml is None:
            raise Exception('Need to pass either yaml or dict')

        self.add_stats()

    def __str__(self):
        pass

    def debug_message(self, s):
        if self.debug:
            print(s)

    def process_hardware_config_dict(self, hardware_dict):
        self.hw_params = hardware_dict

    def process_hardware_config(self, hardware_yaml):
        self.hw_params = AttrDict(yaml.safe_load(open(hardware_yaml)))

    def run_model(self):
        pass

    def conv2d_dw(self, layer_attr, hw_params):
        pass

    def conv2d_pw(self, layer_attr, hw_params):
        pass

    def add_stats(self):
        self.stat_list = ['in_dma_act', 'in_dma_wgt', 'out_dma_act',
                          'mem_wgt', 'mem_in_act', 'mem_out_act',
                          'mem_partial_product',
                          'padd_total', 'theoretical_max_padd_total', 'max_padd_ops_required_per_batch',
                          'cumm_mac_cycles', 'theoretical_max_mac_cycles',
                          'mac_units_available', 'total_mac_units','cycles_max_per_batch',
                          'is_dma_cycle_selected', 'is_mac_cycle_selected', 'mac_cycles', 'cycles_total']
        self.layer_names = []

        for stat in self.stat_list:
            self.stats[stat] = [0]*self.net.num_layers


    def insert_max_stats(self, key, idx, new_stat):
        prev_stat = self.stats[key][idx]
        self.stats[key][idx] = max(prev_stat, new_stat)


    def print_stats(self):
        # write params
        params = self.hw_params.HWConfig
        param_names = ''
        param_list = [0] * self.net.num_layers
        file_name_suffix = ''
        idx = 0
        for k, v in params.items():
            param_names += '_' + str(k)
            file_name_suffix += '_' + str(v)
            if idx >= self.net.num_layers:
                param_list.append(v)
            else:
                param_list[idx] = v
            idx += 1

        param_string = param_names + ',' + ','.join([str(i) for i in param_list])

        # write stats
        dir = self.result_dir + self.model_name

        if not os.path.exists(dir):
            os.makedirs(dir)
        filename =  dir + '/'+ self.model_name + '_' + self.__str__() + file_name_suffix +'.csv'
        with open(filename, 'w') as f:
            for key,value in self.stats.items():
                if self.debug:
                    print(key, value)
                val_list = ','.join([str(i) for i in value])
                row = '{},{}\n'.format(key, val_list)
                f.write(row)

            if self.debug:
                row = ','.join([str(i) for i in self.layer_names])
                f.write(row)

            f.write(param_string)
            f.close()
        return int(sum(self.stats['cycles_total']))



    def load_hw_params_depthwise(self):
        params = self.hw_params.HWConfig
        hw_params = AttrDict({'cx': params.cx, 'wx': params.wx, 'hx': params.hx,
                              'dma_cycles': params.dma_cycles, 'mac_cycles': params.mac_cycles,
                              'padd_cycles': params.padd_cycles
                              })
        return hw_params

    def load_hw_params_pointwise(self, is_first):
        params = self.hw_params.HWConfig
        # if is_first:
        hw_params = AttrDict({'cxx': params.cxx, 'wxx': params.wxx, 'hxx': params.hxx, 'fx': params.fx,
                              'padd_unit': params.padd_unit, 'dma_cycles': params.dma_cycles,
                             'mac_cycles': params.mac_cycles, 'padd_cycles':params.padd_cycles
                              })
        # else:
        #     hw_params = AttrDict({'cxx': params.cxx2, 'wxx': params.wxx2, 'hxx': params.hxx2, 'fx': params.fx2,
        #                           'padd_unit': params.padd_unit2
        #                           })
        return hw_params

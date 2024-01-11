# from dnn_schedules.hwc.hwc_schedule import HWCSchedule
from tools.hw_simulator.schedule.dnn_schedules.hwc.hwc_schedule import HWCSchedule
import math


class HWCFSchedule(HWCSchedule):

    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)
        self.MAC_HW_BIT=8

    def __str__(self):
        return 'hwcf_schedule'

    def run_model(self):
        # orig_idx=0
        for layer_name, layer_attr in self.net.layers.items():
            # orig_idx += 1
            if layer_attr.attr_type == 'DW':
                # self.stats['orig_idx'][layer_attr.layer_idx] = orig_idx - 1
                dw_layer_hw_params = self.load_hw_params_depthwise()
                self.conv2d_dw(layer_attr, dw_layer_hw_params)
                self.layer_names.append(layer_attr.name)
            if layer_attr.attr_type == 'PW' or layer_attr.attr_type == '3d':
                # self.stats['orig_idx'][layer_attr.layer_idx] = orig_idx - 1
                pw_layer_hw_params = self.load_hw_params_pointwise(True)
                self.conv2d(layer_attr, pw_layer_hw_params)
                self.layer_names.append(layer_attr.name)
        return

    # pointwise only stores partial products of output  in memory.
    # This implies input activations will be streamed multiple times. Hence, high DMA operations
    # Note this is a bad design, better idea is to store entire input/output activation
    # TODO: what to do with vector adds -- need to have enough to not stall pipeline
    def conv2d(self, layer_attr, hw_params, init_start_cout_idx=0, init_start_hout_idx=0, init_start_wout_idx=0,
                  is_cross_layer=False, is_first_layer=True, is_last_layer=True):

        if not is_cross_layer:
            mac_units = hw_params.cxx * (hw_params.wxx - layer_attr.Ky + 1) * hw_params.fx
            self.insert_max_stats('mac_units_available', layer_attr.layer_idx, mac_units)
            self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)

        self.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
        wgt_volume = layer_attr.Cout * layer_attr.Cin * layer_attr.Ky * layer_attr.Kx
        if not is_cross_layer:
            self.stats['cycles_total'][layer_attr.layer_idx] += hw_params.dma_cycles*wgt_volume*layer_attr.bit_weights/self.BYTESIZE
            self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1

            self.debug_message('inDMA wgts [0:{}][0:{}]'.format(layer_attr.Cout - 1, layer_attr.Cin - 1))
            self.stats['in_dma_wgt'][layer_attr.layer_idx] = wgt_volume
            self.insert_max_stats('mem_wgt', layer_attr.layer_idx, wgt_volume)

        start_hout_idx = init_start_hout_idx
        for hin in range(0, layer_attr.Hin, hw_params.hxx):
            # Adjust hin indices which will be used from previous convolutions
            # Note: no such assumption is made for 'w' dimension
            if hin != 0:
                hin = hin - layer_attr.Kx + 1

            end_hin_idx = min(hin + hw_params.hxx, layer_attr.Hin) - 1
            num_hin = end_hin_idx - hin + 1
            if num_hin < layer_attr.Kx:
                num_h_convs = 1
            else:
                # In case of last values -- need to add padding information,
                #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
                num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

            end_hout_idx = start_hout_idx + num_h_convs - 1
            num_hout = end_hout_idx - start_hout_idx + 1

            start_wout_idx = init_start_wout_idx
            for win in range(0, layer_attr.Win, hw_params.wxx):
                if win != 0:
                    win = win - layer_attr.Ky + 1

                end_win_idx = min(win + hw_params.wxx, layer_attr.Win) - 1
                num_win = end_win_idx - win + 1
                if num_win < layer_attr.Ky:
                    num_w_convs = 1
                else:
                    # note: # macs connections will differ for stride = 2
                    num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

                end_wout_idx = start_wout_idx + num_w_convs - 1
                num_wout = end_wout_idx - start_wout_idx + 1
                for cin in range(0, layer_attr.Cin, hw_params.cxx):
                    self.conv2d_block(cin, win, hin, start_wout_idx, start_hout_idx, layer_attr, hw_params,
                                         is_cross_layer,
                                         is_first_layer, init_start_cout_idx= init_start_cout_idx)
                # end cin

                if (is_cross_layer and is_last_layer) or not is_cross_layer:
                    self.stats['out_dma_act'][layer_attr.layer_idx] += layer_attr.Cout * num_wout * num_hout
                    self.debug_message('outDMA op_act[0:{}][{}:{}][{}:{}]'.format(layer_attr.Cout - 1,
                                                                                  start_wout_idx, end_wout_idx,
                                                                                  start_hout_idx, end_hout_idx))
                    self.insert_max_stats('mem_out_act', layer_attr.layer_idx, layer_attr.Cout * num_wout * num_hout)
                if is_cross_layer and not is_last_layer:
                    self.debug_message('mem_out_act op_act[0:{}][{}:{}][{}:{}]'.format(layer_attr.Cout - 1,
                                                                                  start_wout_idx, end_wout_idx,
                                                                                  start_hout_idx, end_hout_idx))
                    self.insert_max_stats('mem_out_act', layer_attr.layer_idx, layer_attr.Cout * num_wout * num_hout)

                self.debug_message('====')
                start_wout_idx = end_wout_idx + 1
            # end w
            start_hout_idx = end_hout_idx + 1
        # end h
        return


    def conv2d_block(self, cin, win, hin, start_wout_idx, start_hout_idx, layer_attr, hw_params,
                        is_cross_layer= False,
                        is_first_layer=True, is_last_layer=True, init_start_cout_idx=0):

        num_macs_wx = hw_params.wxx - layer_attr.Ky + 1
        # ------  h parameter calculations
        end_hin_idx = min(hin + hw_params.hxx, layer_attr.Hin) - 1
        num_hin = end_hin_idx - hin + 1
        if num_hin < layer_attr.Kx:
            num_h_convs = 1
        else:
            num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

        end_hout_idx = start_hout_idx + num_h_convs - 1
        num_hout = end_hout_idx - start_hout_idx + 1
        # ------  w parameter calculations
        end_win_idx = min(win + hw_params.wxx, layer_attr.Win) - 1
        num_win = end_win_idx - win + 1
        if num_win < layer_attr.Ky:
            num_w_convs = 1
        else:
            num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

        end_wout_idx = start_wout_idx + num_w_convs - 1
        num_wout = end_wout_idx - start_wout_idx + 1

        # ------ c parameter calculations
        start_cout_idx = cin
        end_cout_idx = min(start_cout_idx + hw_params.cxx, layer_attr.Cin) - 1
        num_cout = end_cout_idx - start_cout_idx + 1
        end_cin_idx = end_cout_idx
        num_cin = num_cout

        self.debug_message(' -- ')
        if (is_cross_layer and is_first_layer) or not is_cross_layer:
            self.debug_message('inDMA ip_act[{}:{}][{}:{}][{}:{}]'.format(cin, end_cin_idx,
                                                                          win, end_win_idx,
                                                                          hin, end_hin_idx
                                                                          ))

            self.stats['in_dma_act'][layer_attr.layer_idx] += num_cin * num_win * num_hin
            cur_in_act_memory = num_cin * num_hin * num_win
            self.insert_max_stats('mem_in_act', layer_attr.layer_idx, cur_in_act_memory)
        num_mac_cycles_all_filters = 0
        for f in range(0, layer_attr.Cout, hw_params.fx):
            end_f_idx = min(f + hw_params.fx, layer_attr.Cout) - 1
            num_f = end_f_idx - f + 1
            # mac utilization
            num_partial_convs = num_f * num_cout * num_w_convs * num_h_convs
            num_mac_cycles_all_filters += num_h_convs*hw_params.mac_cycles* math.ceil(layer_attr.bit_activations/self.MAC_HW_BIT)
            self.stats['cumm_mac_cycles'][layer_attr.layer_idx] += num_partial_convs*hw_params.mac_cycles
            self.stats['theoretical_max_mac_cycles'][
                layer_attr.layer_idx] += hw_params.fx * hw_params.cxx * num_macs_wx * num_h_convs*hw_params.mac_cycles
            # self.debug_message('partial_out_act = C[{}:{}]cin[{}:{}]'
            #                    'W[{}:{}]H[{}:{}]'.format(f, end_f_idx, start_cout_idx, end_cout_idx,
            #                                              start_wout_idx, end_wout_idx,
            #                                              start_hout_idx, end_hout_idx))
        # end f
        # partial adds
        # For DW+PW --> init_start_cout_idx != 0  -- fine grain cross layer parallelism
        # For 2nd pointwise in PDP --> cin != 0
        # For per layer --> cin != 0

        total_padd_cycles =0
        if cin != 0 or (is_cross_layer and (init_start_cout_idx != 0)):
            # since streaming in hx only cx*wx *fx mac happens which are reduced to wx*fx
            # which are partial products
            self.insert_max_stats('max_padd_ops_required_per_batch', layer_attr.layer_idx,
                                  layer_attr.Cout * num_wout)
            self.stats['padd_total'][layer_attr.layer_idx] += layer_attr.Cout * num_wout * num_hout
            total_padd_cycles = layer_attr.Cout * num_wout * \
                                math.ceil(num_hout/hw_params.padd_unit)*hw_params.padd_cycles
            self.stats['theoretical_max_padd_total'][layer_attr.layer_idx] += \
                layer_attr.Cout * hw_params.wxx * hw_params.hxx
            self.insert_max_stats('mem_partial_product', layer_attr.layer_idx,
                                  layer_attr.Cout * num_wout * num_hout)

        # cycles information
        current_batch_cycles = num_mac_cycles_all_filters + total_padd_cycles
        self.stats['mac_cycles'][layer_attr.layer_idx] += current_batch_cycles
        dma_cycles = num_cin * num_win * num_hin * hw_params.dma_cycles*layer_attr.bit_activations/self.BYTESIZE

        self.insert_max_stats('cycles_max_per_batch', layer_attr.layer_idx, current_batch_cycles)

        if not is_cross_layer or (is_cross_layer and is_first_layer):
            # if dma cost is higher then add dma cycles
            if dma_cycles > current_batch_cycles:
                self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
                self.stats['cycles_total'][layer_attr.layer_idx] += dma_cycles
            else:
                self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
                self.stats['cycles_total'][layer_attr.layer_idx] += current_batch_cycles
        else:
            self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
            self.stats['cycles_total'][layer_attr.layer_idx] += current_batch_cycles

        return

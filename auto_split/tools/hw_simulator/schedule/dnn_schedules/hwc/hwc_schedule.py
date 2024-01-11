# from dnn_schedules.schedule import Schedule
from tools.hw_simulator.schedule.dnn_schedules.schedule import Schedule

class HWCSchedule(Schedule):
    def __init__(self, net, model_name, result_dir, verbose,  hardware_yaml=None, hardware_dict=None):
        super().__init__(net, model_name, result_dir, verbose,  hardware_yaml, hardware_dict)

    def __str__(self):
        return 'hwc_schedule'

    def run_model(self):
        pass

    # Only works for stride =1 and padding = 'valid'
    def conv2d_dw(self, layer_attr, hw_params,
                  init_start_hout_idx=0, init_start_wout_idx=0,
                  is_cross_layer=False, is_first_layer=True, is_last_layer=True):

        if not is_cross_layer:
            mac_units = hw_params.cx * (hw_params.wx - layer_attr.Kx + 1)
            self.insert_max_stats('mac_units_available', layer_attr.layer_idx, mac_units)
            self.insert_max_stats('total_mac_units', layer_attr.layer_idx, mac_units)

        self.debug_message('{} {} {}'.format(layer_attr.layer_idx, layer_attr.name, layer_attr.attr_type))
        # Bring in all weights and store it on chip
        total_weights = int(layer_attr.Cin * layer_attr.Depth_multiplier * layer_attr.Kx * layer_attr.Ky)
        if not is_cross_layer:
            self.debug_message('inDMA wgts[{}:{}]'.format(0, total_weights-1))
            self.stats['in_dma_wgt'][layer_attr.layer_idx] += total_weights
            self.stats['cycles_total'][layer_attr.layer_idx] += hw_params.dma_cycles * total_weights*layer_attr.bit_weights/self.BYTESIZE
            self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1

        start_hout_idx = init_start_hout_idx
        for hin in range(0, layer_attr.Hin, hw_params.hx):
            # Adjust hin indices which will be used from previous convolutions
            if hin != 0:
                hin = hin - layer_attr.Kx + 1

            num_hin = min(hin + hw_params.hx, layer_attr.Hin) - hin
            if num_hin < layer_attr.Kx:
                num_h_convs = 1
            else:
                # In case of last values -- need to add padding information,
                #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
                num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

            end_hout_idx = start_hout_idx + num_h_convs - 1

            self.debug_message('=====')
            start_wout_idx = init_start_wout_idx
            for win in range(0, layer_attr.Win, hw_params.wx):
                if win != 0:
                    win = win - layer_attr.Ky + 1

                num_win = min(win + hw_params.wx, layer_attr.Win) - win
                # note: # macs connections will differ for stride = 2
                if num_win < layer_attr.Ky:
                    num_w_convs = 1
                else:
                    num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

                end_wout_idx = start_wout_idx + num_w_convs - 1
                for cin in range(0, layer_attr.Cin, hw_params.cx):
                    self.conv2d_dw_block(layer_attr, hw_params, cin, win, hin, start_hout_idx, start_wout_idx,
                                         is_cross_layer, is_first_layer,
                                         is_last_layer)
                # end cin
                start_wout_idx = end_wout_idx + 1
                self.debug_message(' --- ')
            # end win
            start_hout_idx = end_hout_idx + 1
        # end hin
        return start_hout_idx, start_wout_idx

    def conv2d_dw_block(self, layer_attr, hw_params, cin, win, hin, start_hout_idx, start_wout_idx,
                        is_cross_layer=False,
                        is_first_layer=True,
                        is_last_layer=True):
        num_macs = hw_params.wx - layer_attr.Ky + 1

        # -- calculate hin params --
        end_hin_idx = min(hin + hw_params.hx, layer_attr.Hin) - 1
        num_hin = end_hin_idx - hin + 1
        # In case of last values -- need to add padding information,
        #  Also num_hin - layer_attr.Kx has to be divisible - This depends on hx and wx values
        if num_hin < layer_attr.Kx:
            num_h_convs = 1
        else:
            num_h_convs = int(num_hin - layer_attr.Kx / layer_attr.Sx) + 1

        end_hout_idx = start_hout_idx + num_h_convs - 1
        # -- calculate win params --
        end_win_idx = min(win + hw_params.wx, layer_attr.Win) - 1
        num_win = end_win_idx - win + 1
        # note: # macs connections will differ for stride = 2
        if num_win < layer_attr.Ky:
            num_w_convs = 1
        else:
            num_w_convs = int((num_win - layer_attr.Ky) / layer_attr.Sy) + 1

        end_wout_idx = start_wout_idx + num_w_convs - 1
        # -- calculate cin params
        start_cout_idx = int(cin * layer_attr.Depth_multiplier)
        end_cout_idx = min(int(start_cout_idx + hw_params.cx * layer_attr.Depth_multiplier),
                           layer_attr.Cin) - 1
        num_cout = end_cout_idx - start_cout_idx + 1
        end_cin_idx = min(cin + hw_params.cx, layer_attr.Cin) - 1
        num_cin = end_cin_idx - cin + 1

        if not is_cross_layer:
            # -- dma weights
            cur_wgt_memory = num_cout * layer_attr.Kx * layer_attr.Ky
            self.insert_max_stats('mem_wgt', layer_attr.layer_idx, cur_wgt_memory)

        if (is_first_layer and is_cross_layer) or not is_cross_layer:
            self.debug_message('inDMA ip_act[{}:{}][{}:{}][{}:{}]'.format(cin, end_cin_idx,
                                                                          win, end_win_idx,
                                                                          hin, end_hin_idx
                                                                          ))
            self.stats['in_dma_act'][layer_attr.layer_idx] += num_cin * (end_win_idx - win + 1) * (
                    end_hin_idx - hin + 1)

            cur_in_act_memory = num_cin * num_hin * num_win
            self.insert_max_stats('mem_in_act', layer_attr.layer_idx, cur_in_act_memory)

        # cycles information
        dma_cycles = num_cin * (end_win_idx - win + 1) * (end_hin_idx - hin + 1)*layer_attr.bit_activations/self.BYTESIZE
        mac_cycles = num_h_convs*hw_params.mac_cycles
        self.stats['mac_cycles'][layer_attr.layer_idx] += mac_cycles

        if (is_first_layer and is_cross_layer) or not is_cross_layer:
            if dma_cycles > num_h_convs:
                self.stats['cycles_total'][layer_attr.layer_idx] += dma_cycles
                self.stats['is_dma_cycle_selected'][layer_attr.layer_idx] += 1
            else:
                self.stats['cycles_total'][layer_attr.layer_idx] += mac_cycles
                self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1
        else:
            self.stats['cycles_total'][layer_attr.layer_idx] += mac_cycles
            self.stats['is_mac_cycle_selected'][layer_attr.layer_idx] += 1

        self.insert_max_stats('cycles_max_per_batch', layer_attr.layer_idx, num_h_convs)
        # mac utilization
        self.stats['cumm_mac_cycles'][layer_attr.layer_idx] += num_w_convs * num_h_convs * num_cin*hw_params.mac_cycles
        self.stats['theoretical_max_mac_cycles'][layer_attr.layer_idx] += num_macs * num_h_convs * \
                                                                          hw_params.cx*hw_params.mac_cycles

        num_wout = end_wout_idx - start_wout_idx + 1
        num_hout = end_hout_idx - start_hout_idx + 1

        if (is_cross_layer and is_last_layer) or not is_cross_layer:
            self.debug_message('outDMA[{}:{}][{}:{}][{}:{}]'.format(start_cout_idx, end_cout_idx,
                                                                    start_wout_idx, end_wout_idx,
                                                                    start_hout_idx, end_hout_idx))

            self.stats['out_dma_act'][layer_attr.layer_idx] += num_cout * num_wout * num_hout
            cur_out_act_memory = num_cout * num_wout * num_hout
            self.insert_max_stats('mem_out_act', layer_attr.layer_idx, cur_out_act_memory)

        elif is_cross_layer and not is_last_layer:
            self.debug_message('mem_out_act[{}:{}][{}:{}][{}:{}]'.format(start_cout_idx, end_cout_idx,
                                                                    start_wout_idx, end_wout_idx,
                                                                    start_hout_idx, end_hout_idx))
            cur_out_act_memory = num_cout * num_wout * num_hout
            self.insert_max_stats('mem_out_act', layer_attr.layer_idx, cur_out_act_memory)

        return start_cout_idx, end_cout_idx

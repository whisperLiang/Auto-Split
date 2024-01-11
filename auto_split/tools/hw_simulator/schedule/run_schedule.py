
import pandas as pd
from net import Net
from dnn_schedules.hwcf.hwcf_schedule import HWCFSchedule
import os
import datetime
from pathlib import Path

if __name__ == '__main__':
    print(datetime.datetime.now())
    model_names = ['CONV2D', 'resnet18', 'conv3d','p', 'pdp', 'pw_dw',  'DepthSeparable', 'Depthwise', 'Pointwise', 'depth_separable_test0', 'mobilenet']
    device = 'cloud'
    root_dir = os.getcwd() + '/'

    #-----
    if device == 'edge':
        hardware_yaml = root_dir +  "/tools/hw_simulator/schedule/params/hardware_config_edge.yaml"
    elif device == 'cloud':
        hardware_yaml = root_dir +  "/tools/hw_simulator/schedule/params/hardware_config_cloud.yaml"

    hardware_dict = None
    verbose = False
    for model_name in model_names[1:2]:
        data_folder = root_dir + 'generated/bitwidth/'+ model_name  + '/'
        Path(data_folder).mkdir(parents=True, exist_ok=True)
        df = pd.read_csv(data_folder + model_name + '_' + device + '.csv')
        # bitwidth_configuration_folder = 'gen_data/bitwidth/' + model_name + '/'
        # bitwidth_filename = bitwidth_configuration_folder + 'CONV.yaml'
        # Load yaml file -> parse overrides: name, bits_weights, bits_activations
        # Add it as an attribute to the load_model.py after "def extract_layer_features"

        net = Net(df)
        result_dir = root_dir +  'generated/results/'
        Path(result_dir).mkdir(parents=True, exist_ok=True)
        # -- HWCF Schedule ---
        schedule = HWCFSchedule(net, model_name, result_dir, verbose, hardware_yaml=hardware_yaml,
                                hardware_dict=hardware_dict)

        schedule.run_model()
        total_cycles = schedule.print_stats()
        print('total cycles: {}'.format(total_cycles))

    print(datetime.datetime.now())

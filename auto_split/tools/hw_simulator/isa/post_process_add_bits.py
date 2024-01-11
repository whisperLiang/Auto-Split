import numpy as np
import csv
from matplotlib import pyplot as plt
import yaml, os
from pathlib import Path
import io
import pandas as pd
from collections import OrderedDict

def main():
    model_names = ['densenet201','resnet18', 'resnet50', 'mobilenet_v2']
    for model_name in model_names[0:1]:
        load_template_yaml(model_name)
        print('processed: {}'.format(model_name))
# Used for hardware simulations
#Input: raw_data/resnet18.csv => adds bit_activation, bit_weights information => Output: bitwidth/resnet18_cloud.csv
def load_template_yaml(model_name):

    # Set paths
    root_dir = os.getcwd() + '/'
    data_folder= root_dir + 'generated/hw_simulator/raw_data/'
    result_folder= root_dir + 'generated/hw_simulator/post_process/'
    Path(result_folder).mkdir(parents=True, exist_ok=True)
    result_df_file = result_folder + model_name + '.csv'

    df = pd.read_csv(data_folder + model_name + '.csv')
    layer_names = df['name']
    len_df = len(layer_names)
    bit_weights = [8]*len_df
    bit_activations = [8]*len_df
    # add bitwidth information to df
    df['bit_weights'] = bit_weights
    df['bit_activations'] = bit_activations
    df.to_csv(result_df_file)









if __name__ == '__main__':
    # Load model
    # Used for hardware simulations
    # Input: raw_data/resnet18.csv => adds bit_activation, bit_weights information => Output: bitwidth/resnet18_cloud.csv
    # 2. Generate csv (layername, bitwidth_wgt, bitwidth_act, mac_cycle, dma_cycle, padd_cycle )
    main()



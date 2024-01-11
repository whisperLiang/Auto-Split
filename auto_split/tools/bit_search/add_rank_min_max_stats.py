
import datetime
import os
from pathlib import Path
import pandas as pd
import yaml
import argparse
import numpy as np
import math

def string_to_int_list(list_str_csv):
    if list_str_csv is None:
        return  None
    elif type(list_str_csv) != str:
        return [list_str_csv]
    else:
        list_str = list_str_csv.split(',')
        list_int = [int(i) for i in list_str]
    return list_int

def get_node_dependency(input_df, rank_df):
    # convert strings to list of ints
    rank_df['pred'] = rank_df['pred'].apply(string_to_int_list)
    rank_df['succ'] = rank_df['succ'].apply(string_to_int_list)

    cur_dnn_idx = 0
    prev_dnn_idx = -1
    dnn_graph = pd.DataFrame(index= input_df.index, columns=['name','rank','pred','cumm_rank' ,'cumm_pred'])
    for rank_idx,rankdf_row in rank_df.iterrows():
        name = rankdf_row['name']
        rank = rankdf_row['rank']
        pred = rankdf_row['pred']
        if name in input_df['name'].unique() :
            # input_idx = input_df[input_df['name'] == name].index[0]
            dnn_graph.iloc[cur_dnn_idx] = {'name': name, 'rank':rank, 'pred': None,
                                    'cumm_rank': [rank], 'cumm_pred':pred}
            prev_dnn_idx = cur_dnn_idx
            cur_dnn_idx = cur_dnn_idx + 1
        elif prev_dnn_idx == -1:
            continue
        else:
            dnn_pred = dnn_graph['cumm_pred'][prev_dnn_idx]
            dnn_pred.extend(pred)

            dnn_rank = dnn_graph['cumm_rank'][prev_dnn_idx]
            dnn_rank.append(rank)

            # Update
            dnn_graph['cumm_rank'][prev_dnn_idx] = dnn_rank
            dnn_graph['cumm_pred'][prev_dnn_idx] = dnn_pred

    # TODO: Delete all nan entries in names
    dnn_graph.dropna(subset = ["name"], inplace=True)
    # There should be no rank in cumm_rank which is also in pred.
    # Thus, remove them from pred column.
    for idx, row in dnn_graph.iterrows():
        cumm_rank = row['cumm_rank']
        pred = row['cumm_pred']


        for r in cumm_rank:
            while r in pred: pred.remove(r)

        dnn_graph['cumm_pred'][idx] = pred

        # Recursively update pred
        for p in pred:
            for i in range(idx-1,-1,-1):
                pred_rank = dnn_graph['rank'][i]
                pred_cumm_rank = dnn_graph['cumm_rank'][i]
                if p in pred_cumm_rank:
                    if dnn_graph['pred'][idx] is None:
                        dnn_graph['pred'][idx] = [pred_rank]
                    elif pred_rank not in dnn_graph['pred'][idx]:
                        dnn_graph['pred'][idx].append(pred_rank)



    return dnn_graph

def convert_names(x):
    return x.split('.wrapped_module')[0]


def clear_constants(layer_pred, constant_ranks_list):
    filter_pred_list = ''
    if type(layer_pred) == str:
        layer_pred = np.asarray(layer_pred.split(','), dtype=int)
        for x in layer_pred:
            if x not in constant_ranks_list:
                if not filter_pred_list:
                    filter_pred_list = str(x)
                else:
                    filter_pred_list = filter_pred_list + ',' + str(x)
            # else:
            #     print(x)

    if not filter_pred_list: filter_pred_list = np.nan

    return filter_pred_list


def preprocess_rank_df(rank_df):

    # -- Remove Constants --
    constant_idx = rank_df['name'].str.contains("Constant")
    constant_df = rank_df[constant_idx]
    blacklist_rank_list = constant_df['rank'].tolist()
    rank_df = rank_df[~constant_idx]

    # For Yolo-spp
    # Remove all ops with 'pred' =nan except the one with rank=0
    # 2. and with 'pred' in blacklist
    for idx, row in rank_df.iterrows():
        layer_rank = row['rank']
        layer_pred = row['pred']

        if pd.isnull(layer_pred) and layer_rank !=0:
            blacklist_rank_list.append(layer_rank)
            rank_df = rank_df.drop(idx)
        elif type(layer_pred) == str:
            layer_pred = np.asarray(layer_pred.split(','), dtype=int)
            if len(layer_pred) == 1 and layer_pred[0] in blacklist_rank_list:
                blacklist_rank_list.append(layer_rank)
                rank_df = rank_df.drop(idx)
        # else:
        #     print()

    #TODO: remove all references of 'Constants' from the rank_df in pred, succ
    if blacklist_rank_list:
        for idx,row in rank_df.iterrows():
            layer_name = row['name']
            layer_rank = row['rank']
            layer_pred = row['pred']
            layer_succ = row['succ']

            filter_pred_list = clear_constants(layer_pred, blacklist_rank_list)
            rank_df.at[idx, 'pred'] = filter_pred_list

            filter_succ_list = clear_constants(layer_succ, blacklist_rank_list)
            rank_df.at[idx, 'succ'] = filter_succ_list



    return rank_df

def update_rank_quant_stats(input_file_name, rank_file, quant_stats_file, result_file_name):
    input_df = pd.read_csv(input_file_name)
    input_df['name'] = input_df['name'].apply(convert_names)
    rank_df = pd.read_csv(rank_file, index_col=0)
    # Added for yolo
    rank_df = preprocess_rank_df(rank_df)
    rank_df.reset_index(inplace=True, drop=True)

    # Merge rank columns 
    # input_df = input_df.merge(rank_df, left_on='name', right_on='name')
    dnn_graph = get_node_dependency(input_df, rank_df)

    # Iterate over each row of input_df and copy rank and pred.
    # Since, the oredr of layers for input_df and dnn_graph can differ.
    input_df = pd.concat([input_df, pd.DataFrame(
        columns=['rank','pred'])])

    for idx, row in dnn_graph.iterrows():
        layer_name = row['name']
        layer_rank = row['rank']
        layer_pred = row['pred']

        matching_idx = input_df[input_df['name'] == layer_name].index[0]
        input_df.at[matching_idx, 'rank'] = layer_rank
        input_df.at[matching_idx, 'pred'] = layer_pred

    # sort input_df based on rank or dnn schedule.
    input_df = input_df.sort_values(by=['rank'])
    input_df= pd.concat([input_df, pd.DataFrame(
        columns=['min','max','avg_min','avg_max','mean', 'std','b'])])

    # Load quant stats yaml file
    qstats_d = None
    with open(quant_stats_file, 'r') as stream:
        try:
            qstats_d = yaml.safe_load(stream)
        except yaml.YAMLError as exc:
            print(exc)
    stream.close()

    for idx, row in input_df.iterrows():
        name = row['name']
        row['min'] = qstats_d[name]['output']['min']
        row['max'] = qstats_d[name]['output']['max']
        row['avg_min']= qstats_d[name]['output']['avg_min']
        row['avg_max'] = qstats_d[name]['output']['avg_max']
        row['mean'] = qstats_d[name]['output']['mean']
        row['std'] = qstats_d[name]['output']['std']
        row['b'] = qstats_d[name]['output']['b']
        input_df.loc[idx] = row

    len_df = len(input_df['name'])
    bit_weights = [8]*len_df
    bit_activations = [8]*len_df
    # add bitwidth information to df
    input_df['bit_weights'] = bit_weights
    input_df['bit_activations'] = bit_activations
    input_df.to_csv(result_file_name)
    # post_process_rank_df = pd.DataFrame(index=input_df.index, columns=['name','rank','pred'])
    post_process_rank_df = input_df[['name','rank','pred']]
    post_process_rank_df.to_csv(result_file_name.split('.csv')[0] + '_post_rank.csv')
    return input_df, post_process_rank_df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet50', type=lambda s: s.lower(),
                        help='add graph information to model stats dataframe')
    args = parser.parse_args()
    print('Running model: {}'.format(args.arch))
    model_name=args.arch

    # model_names = ['CONV2D', 'resnet18', 'resnet50', 'mobilenet_v2', 'densenet201']
    device = 'cloud'
    root_dir = os.getcwd() + '/'

    # input_file_name = root_dir + 'generated/hw_simulator/post_process/' + model_name + '.csv'
    input_file_name = root_dir + 'data/' + model_name + '/post_prepare_model_df.csv'
    rank_file = root_dir + 'data/' + model_name + '/layer_ranks.csv'
    quant_stats_file = root_dir + 'data/' + model_name + '/quant_stats_after_prepare_model.yaml'
    result_dir = root_dir + 'generated/hw_simulator/post_process2/'
    Path(result_dir).mkdir(parents=True, exist_ok=True)
    result_file_name = result_dir + model_name + '.csv'
    input_df, post_process_rank_df = update_rank_quant_stats(input_file_name, rank_file, quant_stats_file, result_file_name)
    print(input_df)
    print(datetime.datetime.now())
import pandas as pd
from ast import literal_eval as make_tuple
from collections import OrderedDict
from attrdict import AttrDict

class Net():

    def __init__(self, df):
        self.layers = AttrDict()
        self.num_layers = len(df)
        self.extract_layer_features(df)

    def extract_layer_features(self, df):
        # layer = AttrDict()
        num_layers = 0
        total_wgt_vol = 0
        for idx in range(0,len(df)):
            name = df['name'][idx]
            type = df['type'][idx]
            if type == 'Conv2d':
                num_layers = idx
                (Kx,Ky) = make_tuple('(0,0)' if pd.isnull(df['kernel_size'][idx]) else df['kernel_size'][idx])
                if Kx == Ky:
                    K = Kx
                # else:
                #     raise Exception("ERROR KX != Ky not supported ")

                (Nin,Cin,Win,Hin) = make_tuple(df['ifm'][idx])
                (Nout, Cout, Wout, Hout) = make_tuple(df['ofm'][idx])
                mac = df['mac'][idx]
                attr_type = df['attr_type'][idx]
                # groups, stride, padding, bias
                groups = df['groups'][idx]
                (Sx, Sy) = make_tuple(df['stride'][idx])
                (Px,Py) = make_tuple(df['padding'][idx])
                Bias = df['bias'][idx]

                if attr_type == '3d' or attr_type == 'PW':
                    wgt_vol = Cout * Kx * Ky * Cin
                elif attr_type == 'DW':
                    wgt_vol = Cin * int(groups/Cin) * Kx * Ky * 1
                else:
                    raise AssertionError('No support for this layer !!!')
                total_wgt_vol += wgt_vol
                # print('{}: wgt_vol: {}'.format(name, wgt_vol))
                self.layers[name] = AttrDict({'name': name, 'layer_idx': idx, 'type': type,
                                              'Kx': Kx, 'Ky': Ky, 'K': K,
                                              'Nin': Nin, 'Cin': Cin, 'Win':Win, 'Hin': Hin,
                                              'Nout': Nout, 'Cout': Cout, 'Wout': Wout, 'Hout': Hout,
                                              'mac': mac, 'attr_type': attr_type,
                                              'Sx': Sx, 'Sy': Sy, 'Px': Px, 'Py': Py, 'Bias': Bias,
                                              'Depth_multiplier': int(groups/Cin), 'ofm_vol':df['ofm_vol'][idx],
                                              # 'dma_cycles':df['dma_cycles'][idx],
                                              # 'mac_cycles': df['mac_cycles'][idx],
                                              # 'padd_cycles': df['padd_cycles'][idx],
                                              'bit_weights': df['bit_weights'][idx],
                                              'bit_activations': df['bit_activations'][idx],
                                              'wgt_vol': wgt_vol
                                              })

            elif type is 'Linear':
                continue
        num_layers +=1
        self.total_wgt_vol = total_wgt_vol
        self.num_layers = num_layers




if __name__ == '__main__':
    # model_names = ['Depthwise']
    # model_names = ['mobilenet']
    model_names = ['efficientnet-b7', 'efficientnet-b3', 'xception', 'wide_resnet50_2', 'densenet161',
                     'resnext50_32x4d',
                     'inception_v3', 'resnet101', 'resnet152', 'efficientnet-b0', 'resnet50',
                     'mobilenet_v2', 'nasnetamobile', 'mnasnet1_0', 'vgg16',
                     'mobilenet', 'googlenet', 'resnet18',
                     'squeezenet1_0', 'alexnet']

    data_folder='./raw_data/benchmarks/'
    input_vol = 224 * 224 * 3
    for model_name in model_names:

        df = pd.read_csv(data_folder + model_name + '.csv')
        net = Net(df)
        layer_idx = 0
        selected_layer_idx = 0
        for key,value in net.layers.items():

            ofm_vol = value['ofm_vol']
            if input_vol >= ofm_vol:
                reduction = (input_vol- (ofm_vol/2.0))/input_vol*100.0
                selected_layer_idx = layer_idx

                # print('{} {}: {}'.format(key, value.attr_type, reduction))

            layer_idx+=1


        print('{} {}'.format(model_name,selected_layer_idx))








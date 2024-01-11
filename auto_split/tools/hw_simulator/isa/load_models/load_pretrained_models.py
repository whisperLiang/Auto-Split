import torch
import torch.nn as nn
import tools.hw_simulator.hw_utility.model_summary as model_summary
import os
from PIL import Image
import pretrainedmodels
import pretrainedmodels.utils as utils
import time
from pathlib import Path

def time_application(input_batch, model):
    start_time = time.time()
    output_logits = model(input_batch)  # 1x1000
    end_time = time.time()
    runtime = end_time-start_time
    print('execution time: {}'.format(runtime))
    return

def generate_dataframe(model, data_file):
    load_img = utils.LoadImage()

    # transformations depending on the model
    # rescale, center crop, normalize, and others (ex: ToBGR, ToRange255)
    tf_img = utils.TransformImage(model)
    root_dir = os.getcwd() + '/'
    imagefile = 'tools/hw_simulator/isa/load_models/dog.jpg'
    path_img = imagefile

    input_img = load_img(path_img)
    input_tensor = tf_img(input_img)  # 3x400x225 -> 3x299x299 size may differ
    input_tensor = input_tensor.unsqueeze(0)  # 3x299x299 -> 1x3x299x299
    # input_0 = torch.autograd.Variable(input_tensor,
    #                                 requires_grad=False)

    input_0 = input_tensor


    # move the input_0 and model to GPU for speed if available
    if torch.cuda.is_available():
        model.eval()
        input_0 = input_0.to('cuda')
        # input_tensor.to('cuda')
        model.to('cuda')
        time_application(input_0, model)
        dummy_input =  torch.randn((1,3,224,224))
        model_list = [module for module in model.modules() if type(module) != nn.Sequential]
        i=0
        for idx, name  in enumerate(model_list):
            class_name = type(name).__name__

            if class_name == 'Conv2d':
                print(i, name )
                i+=1
            else:
                print(class_name)


        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)


if __name__ == '__main__':

# broken moedels
# 'nasnetalarge',
# ['fbresnet152', 'bninception', 'resnext101_32x4d', 'resnext101_64x4d', 'inceptionv4', 'inceptionresnetv2'

    # pretrainedmodel_list = ['polynet','fbresnet152','resnet50','se_resnext50_32x4d',
    #  'se_resnext101_32x4d','pnasnet5large','nasnetamobile', 'xception']
    # print(pretrainedmodels.model_names)
    model_name = 'xception'
    root_dir = os.getcwd() + '/'
    model = pretrainedmodels.__dict__[model_name](num_classes=1000)
    data_folder = root_dir + 'generated/hw_simulator/raw_data/'
    Path(data_folder).mkdir(parents=True, exist_ok=True)

    data_file = data_folder + model_name + '.csv'
    # data_file = '../data/' + model_name + '.csv'
    generate_dataframe(model, data_file)
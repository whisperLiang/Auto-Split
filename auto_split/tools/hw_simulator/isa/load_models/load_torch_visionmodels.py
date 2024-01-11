from PIL import Image
from torchvision import transforms
import torchvision.models as models
import timeit
import torch
import tools.hw_simulator.hw_utility.model_summary as model_summary
import os
from pathlib import Path
import time

def time_application(input_batch, model):
    start_time = time.time()
    output = model(input_batch)
    end_time = time.time()
    runtime = end_time-start_time
    print('execution time: {}'.format(runtime))
    return



def generate_dataframe(model, data_file, imagefile):
    input_image = Image.open(imagefile)
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    input_tensor = preprocess(input_image)

    input_batch = input_tensor.unsqueeze(0) # create a mini-batch as expected by the model
    time_application(input_batch,model)
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        input_batch = input_batch.to('cuda')
        model.to('cuda')
        output = model(input_batch)

        # x = summary(model, (3, 224, 224))
        dummy_input =  torch.randn((1,3,224,224))
        df = model_summary.model_performance_summary(model, dummy_input, 1)

        df.to_csv(data_file)

        print(df)

if __name__ == '__main__':

    # # load model
    # torchvision_model_names = ['squeezenet1_0', 'resnet18', 'alexnet', 'vgg16', 'densenet161',
    #                'inception_v3', 'googlenet', 'shufflenet_v2_x1_0', 'mobilenet_v2',
    #                'resnext50_32x4d', 'wide_resnet50_2', 'wide_resnet50_2', 'wide_resnet50_2', 'mnasnet1_0']
    #
    # for model_name in torchvision_model_names[0]:
    #     model_name = 'mobilenet_v2'
    #     model_name = 'resnext50_32x4d'
        # model_list = [l for l in model.modules()]
    model_names = ['densenet201','resnet18', 'resnet50', 'mobilenet_v2', 'mnasnet1_0']
    for model_name in model_names[4:5]:
        str = 'models.' + model_name + '(pretrained=True)'
        model = eval(str)
        model.eval()

        # set paths
        root_dir = os.getcwd() + '/'
        data_folder = root_dir + 'generated/hw_simulator/raw_data/'
        Path(data_folder).mkdir(parents=True, exist_ok=True)

        data_file = data_folder + model_name + '.csv'
        imagefile= root_dir + 'tools/hw_simulator/isa/load_models/dog.jpg'

        generate_dataframe(model,data_file,imagefile)

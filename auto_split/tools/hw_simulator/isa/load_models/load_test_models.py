import torch
import tools.hw_simulator.hw_utility.model_summary as model_summary
from tools.hw_simulator.isa.models.conv_example import CONV2D
import os
from pathlib import Path

def generate_dataframe(model, data_file,cin,win,hin):
    # move the input and model to GPU for speed if available
    if torch.cuda.is_available():
        model.to('cuda')
        dummy_input =  torch.randn((1,cin,win,hin))
        df = model_summary.model_performance_summary(model, dummy_input, 1)
        df.to_csv(data_file)
        print(df)


if __name__ == '__main__':

        cin=3
        cout = 4
        win= 4
        hin = 4

        model_name = 'CONV2D'
        model = CONV2D(Cin=cin,K=3, Cout=cout)
        model.eval()
        root_dir = os.getcwd() + '/'
        data_folder = root_dir + 'generated/raw_data/'
        Path(data_folder).mkdir(parents=True, exist_ok=True)

        data_file = data_folder + model_name + '.csv'
        generate_dataframe(model,data_file, cin,win,hin)

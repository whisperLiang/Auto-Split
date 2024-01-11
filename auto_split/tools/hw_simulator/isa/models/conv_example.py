import torch
from torch import optim, nn
import torch.nn.functional as F


class CONV2D(nn.Module):
    def __init__(self,Cin=3,K=3, Cout=4):
        super(CONV2D, self).__init__()
        self.conv1 = nn.Conv2d(Cin, Cout,
                               kernel_size=K, bias=False, padding=0,stride=1)

    def forward(self, x):
        return F.relu(self.conv1(x))
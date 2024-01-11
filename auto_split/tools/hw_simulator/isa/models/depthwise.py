import torch
from torch import optim, nn
import torch.nn.functional as F


class Depthwise(nn.Module):
    def __init__(self,Cin=10,K=3,depth_multiplier=1):
        super(Depthwise, self).__init__()
        self.conv1 = nn.Conv2d(Cin, depth_multiplier*Cin,
                               kernel_size=K, groups=Cin,bias=False, padding=0,stride=1)

    def forward(self, x):
        return F.relu(self.conv1(x))



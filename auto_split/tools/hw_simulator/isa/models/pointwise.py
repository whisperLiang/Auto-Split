import torch
from torch import optim, nn
import torch.nn.functional as F


class Pointwise(nn.Module):
    def __init__(self,Cin=4,K=1, Cout=10):
        super(Pointwise, self).__init__()
        self.conv1 = nn.Conv2d(Cin, Cout,
                               kernel_size=K, bias=False, padding=0,stride=1)

    def forward(self, x):
        return F.relu(self.conv1(x))



import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torchvision.models as models
from torch.autograd import Variable


class PWDW(nn.Module):
    def __init__(self, _cin=3, _cout=6, _stride=1):
        super(PWDW, self).__init__()


        def conv_dw(inp, oup, stride):
            return nn.Sequential(
                nn.Conv2d(inp, oup, kernel_size=1, stride=1, padding=0, bias=False),
                nn.Conv2d(oup, oup, kernel_size=3, stride=stride, padding=0, groups=oup, bias=False)

            )

        self.model = nn.Sequential(
            conv_dw(_cin, _cout, _stride))

    def forward(self, x):
        x = self.model(x)
        return x



def speed(model, name):
    t0 = time.time()
    input = torch.rand(1, 3, 224, 224)
    input = Variable(input, volatile=True)
    t1 = time.time()

    model(input)
    t2 = time.time()

    model(input)
    t3 = time.time()

    print('%10s : %f' % (name, t3 - t2))


if __name__ == '__main__':
    # cudnn.benchmark = True # This will make network slow ??
    ds = PWDW(_cin=3, _cout=6, _stride=1)
    speed(ds, 'Depth separable')
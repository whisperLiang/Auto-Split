import torch
from torchvision.models import resnet50
from thop import profile

model = resnet50()
input = torch.randn(1, 3, 224, 224)
macs, params = profile(model, inputs=(input, ))
print(' ')
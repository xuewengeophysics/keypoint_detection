import torch
import torchvision.models as models
import numpy as np
import torch.nn.functional as F

input_data = torch.randn(1,3,256, 256)
def resnet_cifar(net, input_data):
    x = net.conv1(input_data)
    x = net.bn1(x)
    x = F.relu(x)
    x = net.layer1(x)
    x = net.layer2(x)
    x = net.layer3(x)
    x = net.layer4[0].conv1(x)  #这样就提取了layer4第一块的第一个卷积层的输出
    #x=x.view(x.shape[0],-1)
    return x

model = models.resnet18()
import ipdb;ipdb.set_trace()
x = resnet_cifar(model,input_data)
print(x.shape)
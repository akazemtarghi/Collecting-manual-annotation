

import math
from collections import OrderedDict
import torch.nn as nn
import torch.nn.init as init
import torch as th
import torch.nn.functional as F
from torch.autograd import Variable

class SPPLayer(nn.Module):

    def __init__(self, num_levels, pool_type='max_pool'):
        super(SPPLayer, self).__init__()

        self.num_levels = num_levels
        self.pool_type = pool_type

    def forward(self, x):
        bs, c, h, w = x.size()
        pooling_layers = []
        for i in range(self.num_levels):
            kernel_size = h // (2 ** i)
            if self.pool_type == 'max_pool':
                tensor = F.max_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            else:
                tensor = F.avg_pool2d(x, kernel_size=kernel_size,
                                      stride=kernel_size).view(bs, -1)
            pooling_layers.append(tensor)
        x = th.cat(pooling_layers, dim=-1)
        return x


class Amir(nn.Module):
    def __init__(self, nclass, spp_level=3):

        super(Amir, self).__init__()

        self.spp_level = spp_level
        self.num_grids = 0
        for i in range(spp_level):
            self.num_grids += 2 ** (i * 2)
        print(self.num_grids)

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 6, kernel_size=3, stride=1, padding=2, bias=True),
            #nn.BatchNorm2d(6),
            #nn.Dropout2d(p=0.2),
            nn.Tanh(),
            nn.MaxPool2d(kernel_size=2))

        self.layer2 = nn.Sequential(
            nn.Conv2d(6, 16, kernel_size=3, stride=1, padding=0, bias=True),
            #nn.BatchNorm2d(16),
            #nn.Dropout2d(p=0.2),
            nn.Tanh())
            #nn.MaxPool2d(kernel_size=2))

        self.layer3 = nn.Sequential(
            nn.Conv2d(18, 36, kernel_size=3, padding=1),
            #nn.MaxPool2d(kernel_size=3),
            nn.Tanh())

        self.layer4 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=3, stride=1),
            nn.Tanh())

        self.spp_layer = SPPLayer(spp_level)

        self.fc = nn.Sequential(

            nn.Linear(3136, 1024),
            nn.Tanh(),
            #nn.Dropout(p=0.25),
            nn.Linear(1024, 1024),
            nn.Tanh(),
            #nn.Dropout(p=0.2),
            nn.Linear(1024, nclass),
            )





    def forward(self, x):
        t = self.layer1(x)
        t = self.layer2(t)
        #t = self.layer3(t)
        #t = self.layer4(t)
        t = t.view(x.size(0), -1)
        #t = self.spp_layer(t)
        t = self.fc(t)

        return t


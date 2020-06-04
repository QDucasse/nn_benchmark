# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Mobilenet architecture in PyTorch
# Taken from https://arxiv.org/abs/1704.04861

import torch
import torch.nn as nn
import torch.nn.functional as F

class DWConvPWConv(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(DWConvPWConv, self).__init__()
        self.dw = nn.Sequential(
            nn.Conv2d(in_planes, in_planes, kernel_size=3, stride=stride, padding=1, groups=in_planes, bias=False),
            nn.BatchNorm2d(in_planes),
            nn.ReLU()
        )
        self.pw = nn.Sequential(
            nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_planes),
            nn.ReLU()
        )

    def forward(self, x):
        out = self.dw(x)
        out = self.pw(out)
        return out


class MobilenetV1(nn.Module):
    '''Mobilenet architecture in PyTorch'''
    # (128,2) means conv planes=128, conv stride=2
    cfg = [(64,1),
           (128,2), (128,1),
           (256,2), (256,1),
           (512,2), (512,1), (512,1), (512,1), (512,1), (512,1),
           (1024,2), (1024,1)]

    def __init__(self, n_classes=10, in_channels=3):
        super(MobilenetV1, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            self._make_layers(in_planes=32),
            nn.AvgPool2d(kernel_size=2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(1024, n_classes)
        )
        self.name = "MobilenetV1"

    def _make_layers(self, in_planes):
        layers = []
        for tuple in self.cfg:
            out_planes, stride = tuple[0], tuple[1]
            layers.append(DWConvPWConv(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.softmax(out,dim=1)

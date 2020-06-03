# -*- coding: utf-8 -*-

# MIT License
#
# Copyright (c) 2017 liukuang
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Mobilenet architecture in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F


class Block(nn.Module):
    '''Depthwise conv + Pointwise conv'''
    def __init__(self, in_planes, out_planes, stride=1):
        super(Block, self).__init__()
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
    # (128,2) means conv planes=128, conv stride=2, by default conv stride=1
    cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]

    def __init__(self, n_classes=10, in_channels=3):
        super(MobilenetV1, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.layers = self._make_layers(in_planes=32)
        self.linear = nn.Linear(1024, n_classes)
        self.name = "MobilenetV1"

    def _make_layers(self, in_planes):
        layers = []
        for x in self.cfg:
            out_planes = x if isinstance(x, int) else x[0]
            stride = 1 if isinstance(x, int) else x[1]
            layers.append(Block(in_planes, out_planes, stride))
            in_planes = out_planes
        return nn.Sequential(*layers)

    def conv_bn(self,in_channels)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layers(out)
        out = F.avg_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return F.softmax(out,dim=1)


# Extracted from Model Zoo:

# class MobilenetV1(nn.Module):
#     def __init__(self,in_channels=3,n_classes=10):
#         super(MobilenetV1, self).__init__()
#         cfg = [64, (128,2), 128, (256,2), 256, (512,2), 512, 512, 512, 512, 512, (1024,2), 1024]
#         self.model = nn.Sequential(
#             self.conv_bn(in_channels, 32, 2),
#             self.conv_dw( 32,  64, 1),
#             self.conv_dw( 64, 128, 2),
#             self.conv_dw(128, 128, 1),
#             self.conv_dw(128, 256, 2),
#             self.conv_dw(256, 256, 1),
#             self.conv_dw(256, 512, 2),
#             self.conv_dw(512, 512, 1),
#             self.conv_dw(512, 512, 1),
#             self.conv_dw(512, 512, 1),
#             self.conv_dw(512, 512, 1),
#             self.conv_dw(512, 512, 1),
#             self.conv_dw(512, 1024, 2),
#             self.conv_dw(1024, 1024, 1),
#             nn.AvgPool2d(kernel_size=2)
#         )
#         self.fc = nn.Linear(1024, n_classes)
#         self.name = "MobilenetV1"
#
#     def conv_bn(self,in_channels, out_channels, stride):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True)
#         )
#
#     def conv_dw(self,in_channels, out_channels, stride):
#         return nn.Sequential(
#             nn.Conv2d(in_channels, in_channels, 3, stride, 1, groups=in_channels, bias=False),
#             nn.BatchNorm2d(in_channels),
#             nn.ReLU(inplace=True),
#
#             nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=False),
#             nn.BatchNorm2d(out_channels),
#             nn.ReLU(inplace=True),
#         )
#
#     def forward(self, x):
#         x = self.model(x)
#         x = x.view(-1, 1024)
#         x = self.fc(x)
#         return x

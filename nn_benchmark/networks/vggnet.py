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

# VGGNet 11/13/16/19 architecture in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

cfg = {
    'VGG11': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG13': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'VGG16': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'VGG19': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}

class VGG(nn.Module):
    '''VGG architectures in PyTorch'''
    def __init__(self, vgg_name, in_channels=3, n_classes=10):
        super(VGG, self).__init__()
        self.features = self._make_layers(cfg[vgg_name], in_channels)
        self.classifier = nn.Linear(512, n_classes)
        self.name = vgg_name

    def _make_layers(self, cfg, in_channels):
        layers = []
        in_channels = in_channels
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                layers += [nn.Conv2d(in_channels, x, kernel_size=3, padding=1),
                           nn.BatchNorm2d(x),
                           nn.ReLU(inplace=True)]
                in_channels = x
        layers += [nn.AvgPool2d(kernel_size=1, stride=1)]
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.features(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.softmax(out,dim=1)


def VGG11(in_channels=3, n_classes=10):
    '''VGG11 configuration'''
    return VGG("VGG11",in_channels=in_channels,n_classes=n_classes)

def VGG13(in_channels=3, n_classes=10):
    '''VGG13 configuration'''
    return VGG("VGG13",in_channels=in_channels,n_classes=n_classes)

def VGG16(in_channels=3, n_classes=10):
    '''VGG16 configuration'''
    return VGG("VGG16",in_channels=in_channels,n_classes=n_classes)

def VGG19(in_channels=3, n_classes=10):
    '''VGG19 configuration'''
    return VGG("VGG19",in_channels=in_channels,n_classes=n_classes)

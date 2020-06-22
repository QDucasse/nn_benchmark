# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# VGGNet 11/13/16/19 architecture in PyTorch
# Taken from https://arxiv.org/abs/1409.1556

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
        '''Create the convolutions repetition based on a given configuration'''
        layers = []
        in_channels = in_channels
        # If the item of the configuration is an 'M' --> MaxPool layer
        # Else --> Conv(out=item) => BatchNorm => ReLU
        for x in cfg:
            if x == 'M':
                layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
            else:
                out_channels = x
                layers += [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, padding=1),
                           nn.BatchNorm2d(out_channels),
                           nn.ReLU(inplace=True)]
                in_channels = out_channels
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


if __name__ == "__main__":
    models = [VGG11, VGG13, VGG16, VGG19]
    for model in models:
        mod = model(in_channels=3)
        x = torch.ones([1, 3, 32, 32], dtype=torch.float32)
        torch.onnx.export(mod, x, "tests/networks/export/"+ mod.name + ".onnx")

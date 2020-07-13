# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# ResNeXt architecture for PyTorch

# WIP

import torch
import torch.nn as nn
import torch.nn.functional as F

class Block(nn.Module):
    '''Grouped convolution block with a shortcut.'''
    expansion = 2

    def __init__(self, in_channels=3, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(group_width)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2   = nn.BatchNorm2d(group_width)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*group_width)
        self.relu3 = nn.ReLU()

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

        self.relu4 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.relu3(self.bn3(self.conv3(out)))
        out += self.shortcut(x)
        out = self.relu4(out)
        return out


class ResNeXt(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=10, in_channels=3):
        super(ResNeXt, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=1, bias=False),
            nn.BatchNorm2d(64),
            self._make_layer(num_blocks[0], 1),
            self._make_layer(num_blocks[1], 2),
            self._make_layer(num_blocks[2], 2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(cardinality*bottleneck_width*8, num_classes)
        )


    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        # Increase bottleneck_width by 2 after each stage.
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, 8)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out


def ResNeXt29_2x64d(n_classes=10, in_channels=3):
    return ResNeXt(n_classes = n_classes, in_channels = in_channels,
                   num_blocks=[3,3,3], cardinality=2, bottleneck_width=64)

def ResNeXt29_4x64d(n_classes=10, in_channels=3):
    return ResNeXt(n_classes = n_classes, in_channels = in_channels,
                   num_blocks=[3,3,3], cardinality=4, bottleneck_width=64)

def ResNeXt29_8x64d(n_classes=10, in_channels=3):
    return ResNeXt(n_classes = n_classes, in_channels = in_channels,
                   num_blocks=[3,3,3], cardinality=8, bottleneck_width=64)

def ResNeXt29_32x4d(n_classes=10, in_channels=3):
    return ResNeXt(n_classes = n_classes, in_channels = in_channels,
                   num_blocks=[3,3,3], cardinality=32, bottleneck_width=4)

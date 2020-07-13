# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# ResNet architecture for PyTorch

# WIP

import torch
import torch.nn as nn
import torch.nn.functional as F

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,stride=1, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

        self.relu2 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu2(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1   = nn.BatchNorm2d(out_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2   = nn.BatchNorm2d(out_channels)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(out_channels, self.expansion *
                               out_channels, kernel_size=1, bias=False)
        self.bn3   = nn.BatchNorm2d(self.expansion*out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion*out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion*out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*out_channels)
            )

        self.relu3 = nn.ReLU()

    def forward(self, x):
        out = self.relu1(self.bn1(self.conv1(x)))
        out = self.relu2(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu3(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, name, n_classes=10, in_channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=3,
                      stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            self._make_layer(block,  64, num_blocks[0], stride=1),
            self._make_layer(block, 128, num_blocks[1], stride=2),
            self._make_layer(block, 256, num_blocks[2], stride=2),
            self._make_layer(block, 512, num_blocks[3], stride=2),
            nn.AvgPool2d(kernel_size = 4)
        )

        self.classifier = nn.Sequential(
            self.linear = nn.Linear(512*block.expansion, n_classes)
        )

        self.name = name

    def _make_layer(self, block, out_channels, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, out_channels, stride))
            self.in_planes = out_channels * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out


def ResNet18(n_classes=10, in_channels=3):
    return ResNet(block       = BasicBlock,
                  num_blocks  = [2, 2, 2, 2],
                  n_classes   = n_classes,
                  in_channels = in_channels,
                  name        = "ResNet18")


def ResNet34(n_classes=10, in_channels=3):
    return ResNet(block       = BasicBlock,
                  num_blocks  = [3, 4, 6, 3],
                  n_classes   = n_classes,
                  in_channels = in_channels,
                  name        = "ResNet34")

def ResNet50(n_classes=10, in_channels=3):
    return ResNet(Block       = Bottleneck,
                  num_blocks  = [3, 4, 6, 3],
                  n_classes   = n_classes,
                  in_channels = in_channels,
                  name        = "ResNet50")


def ResNet101(n_classes=10, in_channels=3):
    return ResNet(block       = Bottleneck,
                  num_blocks  = [3, 4, 23, 3],
                  n_classes   = n_classes,
                  in_channels = in_channels,
                  name        = "ResNet101")


def ResNet152(n_classes=10, in_channels=3):
    return ResNet(block       = Bottleneck,
                  num_blocks  = [3, 8, 36, 3],
                  n_classes   = n_classes,
                  in_channels = in_channels,
                  name        = "ResNet152")

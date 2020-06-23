# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# LeNet5 architecture in PyTorch
# Taken from http://yann.lecun.com/exdb/publis/pdf/lecun-01a.pdf

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class LeNet5(nn.Module):
    '''LeNet5 architecture in PyTorch'''
    def __init__(self, n_classes=10, in_channels=3):
        super(LeNet5, self).__init__()

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5, stride=1),
            nn.Tanh(),
            nn.AvgPool2d(kernel_size=2),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=5, stride=1),
            nn.Tanh()
        )

        self.classifier = nn.Sequential(
            nn.Linear(in_features=120, out_features=84),
            nn.Tanh(),
            nn.Linear(in_features=84, out_features=n_classes),
        )

        self.name = "LeNet5"

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.softmax(out, dim=1)

if __name__ == "__main__":
    mod = LeNet5(in_channels=3)
    x = torch.ones([1, 3, 32, 32], dtype=torch.float32)
    torch.onnx.export(mod, x, "tests/networks/export/"+ mod.name + ".onnx")

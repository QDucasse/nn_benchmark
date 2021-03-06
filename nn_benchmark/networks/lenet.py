# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# LeNet architecture in PyTorch

import torch
import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    '''LeNet architecture in PyTorch'''
    def __init__(self, n_classes=10, in_channels=1):
        super(LeNet, self).__init__()
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=6, kernel_size=5, stride=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Linear(16*5*5, 120),
            nn.Linear(120, 84),
            nn.Linear(84, n_classes)
        )

        self.name = "LeNet"

    def forward(self, x):
        out = self.feature_extractor(x)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return F.softmax(out, dim=1)

if __name__ == "__main__":
    mod = LeNet(in_channels=3)
    x = torch.ones([1, 3, 32, 32], dtype=torch.float32)
    torch.onnx.export(mod, x, "tests/networks/export/"+ mod.name + ".onnx")

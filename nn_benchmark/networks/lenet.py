# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Two Conv-2D followed by two Fully-Connected

import torch.nn as nn
import torch.nn.functional as F

class LeNet(nn.Module):
    '''LeNet architecture'''

    def __init__(self,in_channels=3):
        super(LeNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 10, kernel_size=3)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=3)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)
        self.name = "LeNet"

    def forward(self, x):
        # Layer 1
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        # Layer 2
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        # Layer 3
        x = x.view(-1, 320)
        x = self.fc1(F.relu(x))
        x = F.dropout(x, training=self.training)
        # Layer 4
        x = self.fc2(x)
        return F.log_softmax(x,dim=1)

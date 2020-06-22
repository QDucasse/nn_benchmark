# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import unittest
import torch.nn as nn

from nn_benchmark.networks.common import equal_conv, equal_avg_pool, equal_linear, equal_relu, equal_hardtanh

class CommonTestCase(unittest.TestCase):

    def test_equal_conv(self):
        conv1 = nn.Conv2d(in_channels  = 1, out_channels = 3, kernel_size  = 3)
        conv2 = nn.Conv2d(in_channels  = 1, out_channels = 3, kernel_size  = 3)
        self.assertTrue(equal_conv(conv1,conv2))


    def test_equal_avgpool(self):
        avgpool1 = nn.AvgPool2d(kernel_size = 2)
        avgpool2 = nn.AvgPool2d(kernel_size = 2)
        self.assertTrue(equal_avg_pool(avgpool1,avgpool2))

    def test_equal_conv(self):
        linear1 = nn.Linear(in_features  = 10, out_features = 30)
        linear2 = nn.Linear(in_features  = 10, out_features = 30)
        self.assertTrue(equal_linear(linear1,linear2))

    def test_equal_relu(self):
        relu1 = nn.ReLU(inplace=True)
        relu2 = nn.ReLU(inplace=True)
        self.assertTrue(equal_relu(relu1,relu2))

    def test_equal_hardtanh(self):
        htanh1 = nn.Hardtanh(inplace=True)
        htanh2 = nn.Hardtanh(inplace=True)
        self.assertTrue(equal_relu(htanh1,htanh2))

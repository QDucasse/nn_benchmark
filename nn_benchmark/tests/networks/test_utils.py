# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import unittest
import torch.nn as nn

from nn_benchmark.networks.quant_utils.quant_layers      import make_quant_conv2d, make_quant_linear, make_quant_avg_pool
from nn_benchmark.networks.quant_utils.quant_activations import make_quant_relu, make_quant_tanh, make_quant_hard_tanh, make_quant_sigmoid

from nn_benchmark.networks.common             import equal_conv, equal_avg_pool, equal_linear, equal_relu, equal_hardtanh
from nn_benchmark.networks.quant_utils.common import equal_quant_conv, equal_quant_avg_pool, equal_quant_linear, equal_quant_relu, equal_quant_hardtanh


class UtilsTestCase(unittest.TestCase):

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

    def test_equal_quant_conv(self):
        conv1 = make_quant_conv2d(bit_width=4,in_channels=3,out_channels=6)
        conv2 = make_quant_conv2d(bit_width=4,in_channels=3,out_channels=6)
        self.assertTrue(equal_quant_conv(conv1,conv2))

    def test_equal_quant_linear(self):
        linear1 = make_quant_linear(bit_width=4,in_channels=100,out_channels=10)
        linear2 = make_quant_linear(bit_width=4,in_channels=100,out_channels=10)
        self.assertTrue(equal_quant_linear(linear1,linear2))

    def test_equal_quant_avgpool(self):
        avgpool1 = make_quant_avg_pool(bit_width=4,kernel_size=3)
        avgpool2 = make_quant_avg_pool(bit_width=4,kernel_size=3)
        self.assertTrue(equal_quant_avg_pool(avgpool1,avgpool2))

    def test_equal_quant_relu(self):
        relu1 = make_quant_relu(bit_width=4)
        relu2 = make_quant_relu(bit_width=4)
        self.assertTrue(equal_quant_relu(relu1,relu2))

    def test_equal_quant_hardtanh(self):
        hardtanh1 = make_quant_relu(bit_width=4)
        hardtanh2 = make_quant_relu(bit_width=4)
        self.assertTrue(equal_quant_relu(hardtanh1,hardtanh2))

    def test_equal_quant_tanh(self):
        tanh1 = make_quant_relu(bit_width=4)
        tanh2 = make_quant_relu(bit_width=4)
        self.assertTrue(equal_quant_relu(tanh1,tanh2))

    def test_equal_quant_sigmoid(self):
        sigmoid1 = make_quant_relu(bit_width=4)
        sigmoid2 = make_quant_relu(bit_width=4)
        self.assertTrue(equal_quant_relu(sigmoid1,sigmoid2))

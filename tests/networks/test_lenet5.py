# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from nn_benchmark.networks        import LeNet5
from nn_benchmark.networks.common import *

import nn_benchmark.tests.networks.test_common as util

class LeNet5TestCase(unittest.TestCase):
    def setUp(self):
        self.model = LeNet5(n_classes=10, in_channels=3)

        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=5, stride=1),
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
            nn.Linear(in_features=84, out_features=10),
        )

        self.image = torch.ones([1, 3, 32, 32], dtype=torch.float32)
        self.post_feature_extractor_image = self.feature_extractor(self.image)
        self.reshaped_image = torch.flatten(self.post_feature_extractor_image,1)
        self.post_classifier_image = self.classifier(self.reshaped_image)
        self.probabilities = F.softmax(self.post_classifier_image, dim=1)

    def test_initialization(self):
        for target_layer, layer in zip(self.feature_extractor, self.model.feature_extractor):
            if isinstance(layer,nn.Conv2d):
                self.assertTrue(equal_conv(target_layer,layer))
            elif isinstance(layer,nn.AvgPool2d):
                self.assertTrue(equal_avg_pool(target_layer,layer))

        for target_layer, layer in zip(self.classifier, self.model.classifier):
            if isinstance(layer,nn.Linear):
                self.assertTrue(equal_linear(target_layer,layer))

    # def test_feature_extractor(self):
    #     self.assertTrue(torch.all(torch.eq(self.post_feature_extractor_image, self.model.feature_extractor(self.image))))
    #
    # def test_classifier(self):
    #     self.assertTrue(torch.all(torch.eq(self.post_classifier_image, self.model.classifier(self.reshaped_image))))
    #
    # def test_forward(self):
    #     self.assertTrue(torch.all(torch.eq(self.probabilities, self.model.forward(self.image))))

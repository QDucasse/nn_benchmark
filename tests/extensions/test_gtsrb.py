# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import unittest

from nn_benchmark.extensions import GTSRB

class GTSRBTestCase(unittest.TestCase):
    def setUp(self):
        self.gtsrb = GTSRB(root="data")

    def test_train_folder(self):
        self.assertEqual('data/GTSRB/train/GTSRB/train', self.gtsrb.train_folder)

    def test_test_folder(self):
        self.assertEqual('data/GTSRB/train/GTSRB/test', self.gtsrb.test_folder)

    def test_archive_train(self):
        self.assertEqual('data/GTSRB/train/GTSRB/archive_train', self.gtsrb.archive_train)

    def test_archive_test(self):
        self.assertEqual('data/GTSRB/train/GTSRB/archive_test', self.gtsrb.archive_test)

    def test_archive_csv(self):
        self.assertEqual('data/GTSRB/train/GTSRB/archive_csv', self.gtsrb.archive_csv)

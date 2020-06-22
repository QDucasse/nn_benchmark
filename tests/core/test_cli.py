# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import unittest
import pytest

from nn_benchmark.core.cli import Parser, CLI, ObjDict

class ObjDictTestCase(unittest.TestCase):
    def setUp(self):
        self.dummy_dict = {'key1': 1, 'key2': '2', 'key3': 'value3'}
        self.obj_dict   = ObjDict(self.dummy_dict)

    def test_get_attr(self):
        self.assertEqual(self.obj_dict.key1, 1)
        self.assertEqual(self.obj_dict.key2, '2')
        self.assertEqual(self.obj_dict.key3, 'value3')

    def test_get_attr_exception(self):
        with pytest.raises(AttributeError) as e:
            print(self.obj_dict.key5)
        assert str(e.value) == "No such attribute: key5"

    def test_set_attr(self):
        self.obj_dict.key1 = 10
        self.obj_dict.key2 = '20'
        self.obj_dict.key3 = 'value30'
        self.obj_dict.key4 = [40]
        self.assertEqual(self.obj_dict.key1, 10)
        self.assertEqual(self.obj_dict.key2, '20')
        self.assertEqual(self.obj_dict.key3, 'value30')
        self.assertEqual(self.obj_dict.key4, [40])

    def test_del_attr(self):
        del(self.obj_dict.key1)
        with pytest.raises(AttributeError) as e:
            print(self.obj_dict.key1)
        assert str(e.value) == "No such attribute: key1"

    def test_del_attr_exception(self):
        with pytest.raises(AttributeError) as e:
            del(self.obj_dict.key5)
        assert str(e.value) == "No such attribute: key5"


class ParserTestCase(unittest.TestCase):
    def setUp(self):
        self.parser = Parser()
        self.cmd_args = [ '--network', 'QuantVGG11',
        '--dataset', 'FASHION-MNIST', '--epochs', '2',
        '--acq', '2', '--weq', '2', '--inq', '8',
        '--onnx', '--visualize', '--dry_run']

        self.parsed_args = {'datadir': './data/', 'experiments': './experiments',
        'dry_run': True, 'log_freq': 10, 'evaluate': False, 'resume': None,
        'detect_nan': False, 'num_workers': 4, 'gpus': '0', 'batch_size': 100,
        'lr': 0.02, 'optim': 'SGD', 'loss': 'CrossEntropy', 'scheduler': 'FIXED',
        'milestones': '100,150,200,250', 'momentum': 0.9, 'weight_decay': 0,
        'epochs': 2, 'random_seed': 1, 'network': 'QuantVGG11', 'pretrained': False,
        'dataset': 'FASHION-MNIST', 'visualize': True, 'acq': 2, 'weq': 2, 'inq': 8,
        'onnx': True }

    def test_none_or_str(self):
        self.assertTrue(self.parser.none_or_str("None") is None)
        self.assertTrue(self.parser.none_or_str("Some string") == "Some string")
        self.assertTrue(self.parser.none_or_str(1) == 1)

    def test_none_or_int(self):
        self.assertTrue(self.parser.none_or_int("None") is None)
        self.assertTrue(self.parser.none_or_int("10") == 10)

    def test_parse_cmd_args(self):
        expected = self.parsed_args
        actual   = ObjDict(self.parser.parse(self.cmd_args).__dict__)
        print(expected)
        print(actual)
        self.assertEqual(expected, actual)

# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import unittest
import sys
import os
import io
import logging

from nn_benchmark.core.logger import Logger, TrainingEpochMeters, EvalEpochMeters, AverageMeter

class AverageMeterTestCase(unittest.TestCase):
    def setUp(self):
        self.avg_meter = AverageMeter()

    def test_initialization(self):
        self.assertEqual(0, self.avg_meter.val)
        self.assertEqual(0, self.avg_meter.avg)
        self.assertEqual(0, self.avg_meter.sum)
        self.assertEqual(0, self.avg_meter.count)

    def test_update(self):
        self.avg_meter.update(10, 2)
        self.assertEqual(10, self.avg_meter.val)
        self.assertEqual(20, self.avg_meter.sum)
        self.assertEqual(2, self.avg_meter.count)
        self.assertEqual(10, self.avg_meter.avg)

class TrainingEpochMetersTestCase(unittest.TestCase):
    def setUp(self):
        self.tr_meter = TrainingEpochMeters()

    def test_initialization(self):
        self.assertEqual(AverageMeter(), self.tr_meter.batch_time)
        self.assertEqual(AverageMeter(), self.tr_meter.data_time)
        self.assertEqual(AverageMeter(), self.tr_meter.losses)
        self.assertEqual(AverageMeter(), self.tr_meter.top1)
        self.assertEqual(AverageMeter(), self.tr_meter.top5)

class EvalEpochMetersTestCase(unittest.TestCase):
    def setUp(self):
        self.ev_meter = EvalEpochMeters()

    def test_initialization(self):
        self.assertEqual(AverageMeter(), self.ev_meter.model_time)
        self.assertEqual(AverageMeter(), self.ev_meter.loss_time)
        self.assertEqual(AverageMeter(), self.ev_meter.losses)
        self.assertEqual(AverageMeter(), self.ev_meter.top1)
        self.assertEqual(AverageMeter(), self.ev_meter.top5)

class LoggerTestCase(unittest.TestCase):
    def setUp(self):
        self.output_dir_path = 'nn_benchmark/tests/core/test_logs/'
        self.logger = Logger(output_dir_path=self.output_dir_path,dry_run=False)
        # Setup Training meters
        self.tr_meter = TrainingEpochMeters()
        attr_dict_tr = self.tr_meter.__dict__
        for i,avg_meter in enumerate(attr_dict_tr):
            attr_dict_tr[avg_meter].val   = i
            attr_dict_tr[avg_meter].avg   = i
            attr_dict_tr[avg_meter].sum   = i
            attr_dict_tr[avg_meter].count = i
        # Setup Eval meter
        self.ev_meter = EvalEpochMeters()
        attr_dict_ev = self.ev_meter.__dict__
        for i,avg_meter in enumerate(attr_dict_ev):
            attr_dict_ev[avg_meter].val   = i
            attr_dict_ev[avg_meter].avg   = i
            attr_dict_ev[avg_meter].sum   = i
            attr_dict_ev[avg_meter].count = i
        # setup log
        self.log = logging.getLogger('log')
        self.log.setLevel(logging.INFO)
        out_hdlr = logging.StreamHandler(sys.stdout)
        out_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        out_hdlr.setLevel(logging.INFO)
        self.log.addHandler(out_hdlr)
        file_hdlr = logging.FileHandler(os.path.join(self.output_dir_path, 'log.txt'))
        file_hdlr.setFormatter(logging.Formatter('%(asctime)s %(message)s'))
        file_hdlr.setLevel(logging.INFO)
        self.log.addHandler(file_hdlr)
        self.log.propagate = False
        # Clear log file for previous logs
        open(os.path.join(self.output_dir_path,'log.txt'), 'w').close()

    def test_initialization(self):
        self.assertEqual('nn_benchmark/tests/core/test_logs/',self.logger.output_dir_path)
        self.assertEqual(self.logger.log, self.log)

    def test_info(self):
        # stdout check
        # from contextlib import redirect_stdout
        # f = io.StringIO()
        # with redirect_stdout(f):
        self.logger.info("Test Info")
        # self.assertTrue(f.getvalue().rstrip().endswith("Test Info"))
        # File check in log.txt
        with open(os.path.join(self.output_dir_path,'log.txt'), 'r') as f:
            first_line = f.readline().rstrip()
        print(first_line)
        self.assertTrue(first_line.endswith("Test Info"))

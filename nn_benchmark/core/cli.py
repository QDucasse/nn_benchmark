# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2019 Xilinx
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# Included in:
# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Command line interface handler

import argparse
import os
import sys
import torch

from nn_benchmark.core.trainer import Trainer


class Parser(argparse.ArgumentParser):
    '''Parser object wrapped around the argparse ArgumentParser including small helper
       methods to help argument handling'''
    def __init__(self):
        super(Parser, self).__init__(description="PyTorch/Brevitas Training")
        self.add_parse_arguments()

    def none_or_str(self,value):
        '''Helper method that returns the value None when the string None is passed,
           otherwise returns the given value'''
        if value == "None":
            return None
        return value

    def none_or_int(self,value):
        '''Helper method that returns the value None when the string None is passed,
           otherwise returns the given value as an integer'''
        if value == "None":
            return None
        return int(value)

    def add_bool_arg(self, name, default):
        '''Helper method to add mutually exclusive boolean'''
        group = self.add_mutually_exclusive_group(required=False)
        group.add_argument("--" + name, dest=name, action="store_true")
        group.add_argument("--no_" + name, dest=name, action="store_false")
        self.set_defaults(**{name: default})

    def add_parse_arguments(self):
        '''Add the actual parse arguments to the arser'''
        # I/O
        self.add_argument("--datadir", default="./data/", help="Dataset location")
        self.add_argument("--experiments", default="./experiments", help="Path to experiments folder")
        self.add_argument("--dry_run", action="store_true", help="Disable output files generation")
        self.add_argument("--log_freq", type=int, default=10)
        # Execution modes
        self.add_argument("--evaluate", dest="evaluate", action="store_true", help="evaluate model on validation set")
        self.add_argument("--resume", dest="resume", type=self.none_or_str,
                            help="Resume from checkpoint. Overrides --pretrained flag.")
        self.add_bool_arg("detect_nan", default=False)
        # Compute resources
        self.add_argument("--num_workers", default=4, type=int, help="Number of workers")
        self.add_argument("--gpus", type=self.none_or_str, default="0", help="Comma separated GPUs")
        # Optimizer hyperparams
        self.add_argument("--batch_size", default=100, type=int, help="batch size")
        self.add_argument("--lr", default=0.02, type=float, help="Learning rate")
        self.add_argument("--optim", type=self.none_or_str, default="SGD", help="Optimizer to use")
        self.add_argument("--loss", type=self.none_or_str, default="CrossEntropy", help="Loss function to use")
        self.add_argument("--scheduler", default="FIXED", type=self.none_or_str, help="LR Scheduler")
        self.add_argument("--milestones", type=self.none_or_str, default='100,150,200,250', help="Scheduler milestones")
        self.add_argument("--momentum", default=0.9, type=float, help="Momentum")
        self.add_argument("--weight_decay", default=0, type=float, help="Weight decay")
        self.add_argument("--epochs", default=1000, type=int, help="Number of epochs")
        self.add_argument("--random_seed", default=1, type=int, help="Random seed")
        # Neural network Architecture
        self.add_argument("--network", default="LeNet", type=str, help="Neural network architecture")
        self.add_argument("--pretrained", action='store_true', help="Load pretrained model")
        # Dataset
        self.add_argument("--dataset", default="MNIST", help="Dataset to train on")
        self.add_argument("--visualize",action="store_true",help="Visualization of the items or predictions")
        # Quantization
        self.add_argument("--acq",default=32,type=int,help="Activation precision (bit-width)")
        self.add_argument("--weq",default=32,type=int,help="Weight precision (bit-width)")
        self.add_argument("--inq",default=32,type=int,help="Input precision (bit-width)")
        # Export as ONNX
        self.add_argument("--onnx",action="store_true",help="Export final model as ONNX")

    def parse(self,args):
        return self.parse_args(args)

class ObjDict(dict):
    def __getattr__(self, name):
        if name in self:
            return self[name]
        else:
            raise AttributeError("No such attribute: " + name)

    def __setattr__(self, name, value):
        self[name] = value

    def __delattr__(self, name):
        if name in self:
            del self[name]
        else:
            raise AttributeError("No such attribute: " + name)

class CLI(object):
    '''Command Line Interface holding the trainer and parser and connecting the two'''
    def __init__(self,cmd_args):
        # Parse the actual command line arguments
        self.parser = Parser()
        args = self.parser.parse(cmd_args)

        # Express the dictionary as an object: dict['key'] --> dict.key
        self.args = ObjDict(args.__dict__)

        # Avoid creating new folders etc.
        if self.args.evaluate:
            self.args.dry_run = True

        # Instanciate the trainer with the parsed arguments
        self.trainer = Trainer(self.args)

    def main(self):
        '''Launch either the evaluation of the configuration or a training routine'''
        if self.args.evaluate:
            with torch.no_grad():
                self.trainer.eval_model()
                if self.args.onnx:
                    self.trainer.export_onnx()
        else:
            self.trainer.train_model()
            if self.args.onnx:
                self.trainer.export_onnx()

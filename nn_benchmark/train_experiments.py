# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import sys
from nn_benchmark.core import ObjDict, Trainer
from nn_benchmark.networks import QuantTFC, QuantCNV, QuantMobilenetV1

if __name__ == "__main__":
    acq_list = [2, 4, 8, 16, 32]
    weq_list = [2, 4, 8, 16, 32]
    inq_list = [8, 8, 8, 32, 32]

    args = {'datadir': './data/', 'experiments': './experiments', 'dry_run': False,
            'log_freq': 10, 'evaluate': False, 'resume': None, 'detect_nan': False,
            'num_workers': 4, 'gpus': '0', 'batch_size': 100, 'lr': 0.01, 'optim': 'ADAM',
            'loss': 'CrossEntropy', 'scheduler': 'STEP', 'milestones': '34,37',
            'momentum': 0.9, 'weight_decay': 0, 'epochs': 40, 'random_seed': 1,
            'network': None, 'pretrained': False, 'dataset': None,
            'visualize': False, 'acq': None, 'weq': None, 'inq': None, 'onnx': False}
    args = ObjDict(args)

    # QuantTFC on Fashion-MNIST
    args.network = "QuantTFC"
    args.dataset = "FASHION-MNIST"
    for acq, weq, inq in zip(acq_list, weq_list, inq_list):
        args.acq = acq
        args.weq = weq
        args.inq = inq
        trainer_tfc = Trainer(args)
        trainer_tfc.train_model()

    # QuantCNV on CIFAR-10
    args.network = "QuantCNV"
    args.dataset = "CIFAR10"
    for acq, weq, inq in zip(acq_list, weq_list, inq_list):
        args.acq = acq
        args.weq = weq
        args.inq = inq
        trainer_cnv = Trainer(args)
        trainer_cnv.train_model()

    # QuantMobilenetV1 on GTSRB
    args.network = "QuantMobilenetV1"
    args.dataset = "GTSRB"
    for acq, weq, inq in zip(acq_list, weq_list, inq_list):
        args.acq = acq
        args.weq = weq
        args.inq = inq
        trainer_mobilenet = Trainer(args)
        trainer_mobilenet.train_model()

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

# Training routine for a network

import os
import random

import time
from datetime import datetime

import torch
import torchvision
import torch.optim         as optim
import torch.nn.functional as F
import matplotlib.pyplot   as plt

import brevitas.onnx as bo

from torch                    import nn
from torch.utils.data         import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision              import transforms
from torchvision.datasets     import MNIST, CIFAR10, FashionMNIST

from nn_benchmark.networks    import LeNet, LeNet5, VGG11, VGG13, VGG16, VGG19, MobilenetV1
from nn_benchmark.networks    import QuantLeNet5, QuantCNV, QuantTFC, QuantMobilenetV1, QuantVGG11, QuantVGG13, QuantVGG16, QuantVGG19

from nn_benchmark.extensions    import SqrHingeLoss, GTSRB
from nn_benchmark.core.logger   import Logger, TrainingEpochMeters, EvalEpochMeters
from nn_benchmark.core.plotter  import Plotter
from nn_benchmark.core.exporter import Exporter

networks = {"LeNet": LeNet,
            "LeNet5": LeNet5,
            "QuantLeNet5": QuantLeNet5,
            "QuantCNV": QuantCNV,
            "QuantTFC": QuantTFC,
            "VGG11": VGG11,
            "VGG13": VGG13,
            "VGG16": VGG16,
            "VGG19": VGG19,
            "QuantVGG11": QuantVGG11,
            "QuantVGG13": QuantVGG13,
            "QuantVGG16": QuantVGG16,
            "QuantVGG19": QuantVGG19,
            "MobilenetV1": MobilenetV1,
            "QuantMobilenetV1": QuantMobilenetV1
}

class Trainer(object):
    def __init__(self,args):
        torch.set_num_threads(4)
        # Save the args as instance variables
        self.args = args
        # Initialize the randomness parameters
        self.init_randomness(args.random_seed)
        # Initialize the device
        self.device = None
        self.init_device()

        # Initialize the dataset
        self.train_loader = None
        self.test_loader  = None
        self.classes      = None
        self.num_classes  = None
        self.in_channels  = None
        self.plot_set     = None
        self.init_dataset(args.dataset,args.datadir,args.batch_size,args.num_workers)

        # Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0

        # Initialize the model
        self.model = None
        self.act_bit_width    = None
        self.weight_bit_width = None
        self.input_bit_width  = None
        self.init_model(args.network,args.resume,args.acq,args.weq,args.inq)

        # Initialize the output directory
        self.output_dir_path = './'
        self.init_output(args.resume)

        # Initialize the ONNX exporter
        self.exporter = Exporter()

        # Initialize the logger
        self.logger = None
        self.init_logger(args.dry_run,args.resume)
        self.logger.info("Training logs of network {} on dataset {} with bitwidths {}, {}, {} (activation, weight, input)".format(
                                        self.args.network, self.args.dataset, self.args.acq, self.args.weq, self.args.inq))

        # Initialize the plotter
        self.plotter = None
        self.init_plotter(self.plot_set, self.model, self.classes)

        # Initialize the optimizer
        self.optimizer = None
        self.init_optim(args.optim,args.resume,args.evaluate)
        #Initialize the loss function
        self.criterion = None
        self.init_loss(args.loss)
        # Initialize the scheduler
        self.scheduler = None
        self.init_scheduler(args.scheduler,args.milestones,args.resume,args.evaluate)


# ==============================================================================
# ========================= INITIALIZATION METHODS =============================
# ==============================================================================

    def init_output(self,resume):
        '''Initializes the output directory of the experiments:
           experiments/<network name>_<timestamp>'''
        experiment_name = "{0}_A{1}W{2}I{3}_{4}".format(self.model.name, str(self.act_bit_width),
                                                        str(self.weight_bit_width), str(self.input_bit_width),
                                                        datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.output_dir_path = os.path.join(self.args.experiments, experiment_name)
        # Goes back to the experiments folder in case of resume
        if self.args.resume:
            self.output_dir_path, _ = os.path.split(resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)


    def init_model(self,network, resume, act_bit_width, weight_bit_width, input_bit_width):
        '''Initializes the network architecture model'''
        is_quantized = network.startswith("Quant")
        builder = networks[network]
        self.act_bit_width    = act_bit_width
        self.weight_bit_width = weight_bit_width
        self.input_bit_width  = input_bit_width
        # Add the quantization parameters if the network is quantized
        if is_quantized:
            self.model = builder(n_classes        = self.num_classes,
                                 in_channels      = self.in_channels,
                                 act_bit_width    = act_bit_width,
                                 weight_bit_width = weight_bit_width,
                                 in_bit_width     = input_bit_width)
        else:
            self.model = builder(n_classes   = self.num_classes,
                                 in_channels = self.in_channels)

        if resume:
            print('Loading model checkpoint at: {}'.format(resume))
            package = torch.load(resume, map_location='cpu')
            model_state_dict = package['state_dict']
            self.model.load_state_dict(model_state_dict)

    def init_logger(self,dry_run,resume):
        '''Initializes the logger with the correct output directory and specify
           if txt files are needed (dry_run=True) or not (dry_run=False)'''
        # Create the checkpoint directory if logs have to be created
        if not dry_run:
            self.checkpoints_dir_path = os.path.join(self.output_dir_path, 'checkpoints')
            if not resume:
                os.mkdir(self.output_dir_path)
                os.mkdir(self.checkpoints_dir_path)
        self.logger = Logger(self.output_dir_path, dry_run)

    def init_plotter(self, plot_set, model, classes):
        '''Initializes the plotter on a plot set with no transformations and a given
           trained model'''
        self.plotter = Plotter(plot_set, model, classes)

    def init_randomness(self,seed):
        '''Set the random seed for PyTorch'''
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_device(self):
        '''Initializes the device (CPU or GPUs)'''
        self.device = "cpu"

    def init_optim(self,optimizer,resume,evaluate):
        '''Initializes the optimizer. Add an optimizer in the if checks.'''
        lr           = self.args.lr
        momentum     = self.args.momentum
        weight_decay = self.args.weight_decay
        if optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=lr,
                                        weight_decay=weight_decay)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=lr,
                                       momentum=momentum,
                                       weight_decay=weight_decay)
        else:
            raise Exception("Unrecognized optimizer {}".format(optim))

        if resume and not evaluate:
            package = torch.load(resume, map_location='cpu')
            self.logger.log.info("Loading optimizer checkpoint")
            if 'optim_dict' in package.keys():
                self.optimizer.load_state_dict(package['optim_dict'])
            if 'epoch' in package.keys():
                self.starting_epoch = package['epoch']
            if 'best_val_acc' in package.keys():
                self.best_val_acc = package['best_val_acc']


    def init_loss(self,loss):
        '''Initializes the loss function. Add an optimizer in the if checks'''
        if loss == "CrossEntropy":
            self.criterion = nn.CrossEntropyLoss()
        elif loss == "SqrHinge":
            self.criterion = SqrHingeLoss()
        # self.criterion = self.criterion.to(device=self.device)
        else:
            raise Exception("Unrecognized loss function {}".format(loss))

    def init_scheduler(self,scheduler,milestones,resume,evaluate):
        '''Initializes the scheduler'''
        # STEP scheduler, will modify the learning rate when the epoch correspond to the <milestones>
        if scheduler == 'STEP':
            milestones = [int(i) for i in milestones.split(',')]
            self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                         milestones=milestones,
                                         gamma=0.1)
        # FIXED scheduler, the learning rate is untouched (or at least given a default decay)
        elif scheduler == 'FIXED':
            self.scheduler = None
        else:
            raise Exception("Unrecognized scheduler {}".format(scheduler))

        if resume and not evaluate and self.scheduler is not None:
            self.scheduler.last_epoch = package['epoch'] - 1

    def init_dataset(self,dataset,datadir,batch_size,num_workers):
        '''Initializes the dataset chosen. Add your dataset in the if checks'''
        transform_plot = transforms.Compose([transforms.Resize((32, 32)),
                                             transforms.ToTensor()])
        # CIFAR10 dataset, colored images (3 channels) of 10 distinct classes
        if dataset == 'CIFAR10':
            transform_train = transforms.Compose(
                                    [transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                            )
            transform_test  = transforms.Compose(
                                    [transforms.ToTensor(),
                                     transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))]
                            )
            builder = CIFAR10
            classes = ['Plane', 'Car', 'Bird', 'Cat', 'Deer', 'Dog', 'Frog', 'Horse', 'Ship', 'Truck']
            in_channels = 3

        # MNIST dataset, handwritten digits separated in 10 classes of 28*28 black and white (1 channel) images
        elif dataset == 'MNIST':
            train_transforms_list = [transforms.Resize((32, 32)),
                                     transforms.ToTensor()]
            transform_train = transforms.Compose(train_transforms_list)
            transform_test  = transform_train
            builder = MNIST
            classes = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9']
            in_channels = 1

        # FashionMNIST dataset, handwritten digits separated in 10 classes of 28*28 black and white (1 channel) images
        elif dataset == 'FASHION-MNIST':
            train_transforms_list = [transforms.Resize((32, 32)),
                                     transforms.ToTensor()]
            transform_train = transforms.Compose(train_transforms_list)
            transform_test  = transform_train
            builder = FashionMNIST
            classes = ['T-shirt/Top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
            in_channels = 1

        # GTSRB dataset, handwritten digits separated in 10 classes of 28*28 black and white (1 channel) images
        elif dataset == 'GTSRB':
            train_transforms_list = [ transforms.Resize((32, 32)),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.1307,), (0.3081,))]
            transform_train = transforms.Compose(train_transforms_list)
            transform_test  = transform_train
            builder = GTSRB
            classes = [ 'Speed limit (20km/h)',                'Speed limit (30km/h)',
                        'Speed limit (50km/h)',                'Speed limit (60km/h)',
                        'Speed limit (70km/h)',                'Speed limit (80km/h)',
                        'End of speed limit (80km/h)',         'Speed limit (100km/h)',
                        'Speed limit (120km/h)',               'No passing',
                        'No passing for heavy vehicles',       'Right-of-way at next intersection',
                        'Priority road',                       'Yield',
                        'Stop',                                'No vehicles',
                        'Heavy vehicles prohibited',           'No entry',
                        'General caution',                     'Dangerous curve to the left',
                        'Dangerous curve to the right',        'Double curve',
                        'Bumpy road',                          'Slippery road',
                        'Road narrows on the right',           'Road work',
                        'Traffic signals',                     'Pedestrians',
                        'Children crossing',                   'Bicycles crossing',
                        'Beware of ice/snow',                  'Wild animals crossing',
                        'End of all speed and passing limits', 'Turn right ahead',
                        'Turn left ahead',                     'Ahead only',
                        'Go straight or right',                'Go straight or left',
                        'Keep right',                          'Keep left',
                        'Roundabout mandatory',                'End of no passing',
                        'End of no passing by heavy vehicles'
                         ]
            in_channels = 3

        else:
            raise Exception("Dataset not supported: {}".format(dataset))

        # Extract the builders for the dataset
        train_set = builder(root=datadir,
                            train=True,
                            download=True,
                            transform=transform_train)
        test_set = builder(root=datadir,
                           train=False,
                           download=True,
                           transform=transform_test)
        plot_set = builder(root=datadir,
                           train=False,
                           download=True,
                           transform=transform_plot)
        # Create the corresponding DataLoaders
        self.train_loader = DataLoader(train_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers)
        self.test_loader = DataLoader(test_set,
                                      batch_size=batch_size,
                                      shuffle=True,
                                      num_workers=num_workers)
        self.plot_set = plot_set
        self.classes     = classes
        self.num_classes = len(classes)
        self.in_channels = in_channels

# ==============================================================================
# ================================= ACCURACY ===================================
# ==============================================================================

    def accuracy(self, output, target, topk=(1,)):
        """Computes the precision@k for the specified values of k"""
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

# ==============================================================================
# =============================== MODEL SAVE ===================================
# ==============================================================================

    def checkpoint_best(self, epoch, name):
        '''Save the checkpoint of the model'''
        best_path = os.path.join(self.checkpoints_dir_path, name)
        self.logger.info("Saving checkpoint model to {}".format(best_path))
        torch.save({
            'state_dict': self.model.state_dict(),
            'optim_dict': self.optimizer.state_dict(),
            'epoch': epoch + 1,
            'best_val_acc': self.best_val_acc,
        }, best_path)

    def export_onnx(self, input_tensor=None, torch_onnx_kwargs={}):
        '''Export the model in ONNX format. A different export is provided is the
           network is a quantized one because the quantizations need to be stored as
           specific ONNX attributes'''
        self.exporter.export_onnx(self.model, self.output_dir_path,
                                  self.act_bit_width, self.weight_bit_width,
                                  self.input_bit_width, self.in_channels,
                                  input_tensor=input_tensor, torch_onnx_kwargs=torch_onnx_kwargs)

# ==============================================================================
# ======================== TRAINING AND EVALUATION =============================
# ==============================================================================

    def train_model(self):
        print("Training")

        if self.args.visualize:
            self.plotter.display_items(5,10)

        # Training starts
        if self.args.detect_nan:
            torch.autograd.set_detect_anomaly(True)

        for epoch in range(self.starting_epoch, self.args.epochs):
            # Set to training mode
            self.model.train()
            self.criterion.train()
            # Init metrics
            epoch_meters = TrainingEpochMeters()
            start_data_loading = time.time()
            for i, data in enumerate(self.train_loader):
                (input, target) = data
                # input = input.to(self.device, non_blocking=True)
                # target = target.to(self.device, non_blocking=True)
                # Measure data loading time --> "Data" in the logger
                epoch_meters.data_time.update(time.time() - start_data_loading)
                # Training batch starts
                start_batch = time.time()
                output = self.model(input)
                loss = self.criterion(output, target)

                # Compute gradient and perform the optimizer step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # Measure overall elapsed time --> "Time" column in the logger
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % int(self.args.log_freq) == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = self.accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(epoch_meters, epoch, i, len(self.train_loader))

                # Training batch ends
                start_data_loading = time.time()

            # Set the learning rate with the scheduler or halves the learning rate every 40 epochs
            if self.scheduler is not None:
                self.scheduler.step()
            else:
                if epoch%40==0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5

            # Perform evaluation
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            # Checkpoint save:
            # If the top 1 average is better the one from the last epoch --> Save under "best.tar"
            # Else --> Save under "checkpoint.tar"
            if not self.args.dry_run:
                if top1avg >= self.best_val_acc:
                    self.best_val_acc = top1avg
                    self.checkpoint_best(epoch, "best.tar")
                else:
                    self.checkpoint_best(epoch, "checkpoint.tar")

                # Save a different model every 10 epochs
                if epoch%10==0:
                    self.checkpoint_best(epoch, "checkpoint_" + str(epoch) + ".tar")

        # Training ends
        if not self.args.dry_run:
            return os.path.join(self.checkpoints_dir_path, "best.tar")

        print("Training completed!")

    def eval_model(self, epoch=None):
        print("Evaluating")

        if self.args.visualize:
            self.plotter.display_predictions(5,10)

        eval_meters = EvalEpochMeters()

        # Switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        for i, data in enumerate(self.test_loader):
            # Compute output
            # -- Time start
            model_time_start = time.time()
            # -- Input processing
            (input, target) = data
            # input = input.to(self.device, non_blocking=True)
            # target = target.to(self.device, non_blocking=True)
            target_var = target
            output = self.model(input)
            # -- Time stop and log
            model_time_end = time.time()
            model_time = model_time_end - model_time_start
            eval_meters.model_time.update(model_time)

            # Compute loss
            # -- Time start
            loss_time_start = time.time()
            # -- Loss processing
            loss = self.criterion(output, target_var)
            # -- Time stop and log
            loss_time_end = time.time()
            loss_time = loss_time_start - loss_time_end
            eval_meters.loss_time.update(loss_time)

            # Get the prediction
            pred = output.data.argmax(1, keepdim=True)
            # Check for equality
            correct = pred.eq(target.data.view_as(pred)).sum()
            # Top 1 prec
            prec1 = 100. * correct.float() / input.size(0)
            # Top 5 prec
            _, prec5 = self.accuracy(output, target, topk=(1, 5))
            # Precision logging
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))

            #Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))

        print("Top 1 average: " + str(eval_meters.top1.avg))
        return eval_meters.top1.avg



if __name__ == "__main__":
    from nn_benchmark.core import ObjDict
    acq = 2
    weq = 2
    inq = 8
    model = "QuantCNV"
    dataset = "GTSRB"
    args = {'datadir': './data/', 'experiments': './experiments', 'dry_run': False,
            'log_freq': 10, 'evaluate': False, 'resume': None, 'detect_nan': False,
            'num_workers': 4, 'gpus': '0', 'batch_size': 100, 'lr': 0.01, 'optim': 'ADAM',
            'loss': 'CrossEntropy', 'scheduler': 'FIXED', 'milestones': '100,150,200,250',
            'momentum': 0.9, 'weight_decay': 0, 'epochs': 2, 'random_seed': 1,
            'network': model, 'pretrained': False, 'dataset': dataset,
            'visualize': False, 'acq': acq, 'weq': weq, 'inq': inq, 'onnx': True}
    args = ObjDict(args)
    trainer = Trainer(args)
    trainer.train_model()

# -*- coding: utf-8 -*-

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

from torch                    import nn
from torch.utils.data         import DataLoader
from torch.optim.lr_scheduler import MultiStepLR
from torchvision              import transforms
from torchvision.datasets     import MNIST, CIFAR10

from nn_benchmark.networks import LeNet
from nn_benchmark.core.logger import Logger, TrainingEpochMeters, EvalEpochMeters


class Trainer(object):
    def __init__(self,args):
        # Save the args as instance variables
        self.args = args
        # Initialize the randomness parameters
        self.init_randomness(args.random_seed)
        # Initialize the device
        self.device = None
        self.init_device()

        # Initialize the model
        self.model = None
        self.init_model(args.network,args.resume)

        # Initialize the output directory
        self.output_dir_path = './'
        self.init_output(args.resume)

        # Initialize the logger
        self.logger = None
        self.init_logger(args.dry_run,args.resume)

        # Initialize the optimizer
        self.optimizer = None
        self.init_optim(args.optim,args.resume,args.evaluate)
        #Initialize the loss function
        self.criterion = None
        self.init_loss(args.loss)
        # Initialize the scheduler
        self.scheduler = None
        self.init_scheduler(args.scheduler,args.milestones,args.resume,args.evaluate)

        # Initialize the dataset
        self.train_loader = None
        self.test_loader  = None
        self.init_dataset(args.dataset,args.datadir,args.batch_size,args.num_workers)

        # Init starting values
        self.starting_epoch = 1
        self.best_val_acc = 0


# ==============================================================================
# ========================= INITIALIZATION METHODS =============================
# ==============================================================================

    def init_output(self,resume):
        '''Initializes the output directory of the experiments:
           experiments/<network name>_<timestamp>'''
        experiment_name = "{0}_{1}".format(self.model.name, datetime.now().strftime('%Y%m%d_%H%M%S'))
        self.output_dir_path = os.path.join(self.args.experiments, experiment_name)
        # Goes back to the experiments folder in case of resume
        if self.args.resume:
            self.output_dir_path, _ = os.path.split(resume)
            self.output_dir_path, _ = os.path.split(self.output_dir_path)

    def init_model(self,network,resume):
        '''Initializes the network architecture model'''
        if network == "LeNet":
            self.model = LeNet()

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


    def init_randomness(self,seed):
        '''Set the random seed for PyTorch'''
        random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

    def init_device(self):
        '''Initializes the device (CPU or GPUs)'''
        self.device = "cpu"

    def init_optim(self,optimizer,resume,evaluate):
        '''Initializes the optimizer'''
        lr           = self.args.lr
        momentum     = self.args.momentum
        weight_decay = self.args.weight_decay
        if optimizer == 'ADAM':
            self.optimizer = optim.Adam(self.model.parameters(),
                                        lr=args.lr,
                                        weight_decay=args.weight_decay)
        elif optimizer == "SGD":
            self.optimizer = optim.SGD(self.model.parameters(),
                                       lr=self.args.lr,
                                       momentum=self.args.momentum,
                                       weight_decay=self.args.weight_decay)
        else:
            raise Exception("Unrecognized optimizer {}".format(optim))

        if resume and not evaluate:
            self.logger.log.info("Loading optimizer checkpoint")
            if 'optim_dict' in package.keys():
                self.optimizer.load_state_dict(package['optim_dict'])
            if 'epoch' in package.keys():
                self.starting_epoch = package['epoch']
            if 'best_val_acc' in package.keys():
                self.best_val_acc = package['best_val_acc']

    def init_loss(self,loss):
        '''Initializes the loss function'''
        if loss == 'CrossEntropy':
            self.criterion = nn.CrossEntropyLoss()
        # self.criterion = self.criterion.to(device=self.device)

    def init_scheduler(self,scheduler,milestones,resume,evaluate):
        '''Initializes the scheduler'''
        if scheduler == 'STEP':
            milestones = [int(i) for i in milestones.split(',')]
            self.scheduler = MultiStepLR(optimizer=self.optimizer,
                                         milestones=milestones,
                                         gamma=0.1)
        elif scheduler == 'FIXED':
            self.scheduler = None
        else:
            raise Exception("Unrecognized scheduler {}".format(scheduler))

        if resume and not evaluate and self.scheduler is not None:
            self.scheduler.last_epoch = package['epoch'] - 1

    def init_dataset(self,dataset,datadir,batch_size,num_workers):
        '''Initializes the dataset chosen'''
        transform_to_tensor = transforms.Compose([transforms.ToTensor()])
        if dataset == 'CIFAR10':
            train_transforms_list = [transforms.RandomCrop(32, padding=4),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.ToTensor()]
            transform_train = transforms.Compose(train_transforms_list)
            builder = CIFAR10

        elif dataset == 'MNIST':
            transform_train = transform_to_tensor
            builder = MNIST
        else:
            raise Exception("Dataset not supported: {}".format(dataset))

        train_set = builder(root=datadir,
                            train=True,
                            download=True,
                            transform=transform_train)
        test_set = builder(root=datadir,
                           train=False,
                           download=True,
                           transform=transform_to_tensor)
        self.train_loader = DataLoader(train_set,
                                       batch_size=batch_size,
                                       shuffle=True,
                                       num_workers=num_workers)
        self.test_loader = DataLoader(test_set,
                                      batch_size=batch_size,
                                      shuffle=False,
                                      num_workers=num_workers)

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

# ==============================================================================
# ======================== TRAINING AND EVALUATION =============================
# ==============================================================================

    def train_model(self):
        print("Training")

        # training starts
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
                target_var = target
                # measure data loading time
                epoch_meters.data_time.update(time.time() - start_data_loading)
                # Training batch starts
                start_batch = time.time()
                output = self.model(input)
                loss = self.criterion(output, target_var)

                # compute gradient and do SGD step
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                # measure elapsed time
                epoch_meters.batch_time.update(time.time() - start_batch)

                if i % int(self.args.log_freq) == 0 or i == len(self.train_loader) - 1:
                    prec1, prec5 = self.accuracy(output.detach(), target, topk=(1, 5))
                    epoch_meters.losses.update(loss.item(), input.size(0))
                    epoch_meters.top1.update(prec1.item(), input.size(0))
                    epoch_meters.top5.update(prec5.item(), input.size(0))
                    self.logger.training_batch_cli_log(epoch_meters, epoch, i, len(self.train_loader))

                # training batch ends
                start_data_loading = time.time()

                # Set the learning rate
            if self.scheduler is not None:
                self.scheduler.step(epoch)
            else:
                # Set the learning rate
                if epoch%40==0:
                    self.optimizer.param_groups[0]['lr'] *= 0.5

            # Perform eval
            with torch.no_grad():
                top1avg = self.eval_model(epoch)

            # checkpoint
            if top1avg >= self.best_val_acc and not self.args.dry_run:
                self.best_val_acc = top1avg
                self.checkpoint_best(epoch, "best.tar")
            elif not self.args.dry_run:
                self.checkpoint_best(epoch, "checkpoint.tar")

        # training ends
        if not self.args.dry_run:
            return os.path.join(self.checkpoints_dir_path, "best.tar")

    def eval_model(self, epoch=None):
        print("Evaluating")

        eval_meters = EvalEpochMeters()

        # switch to evaluate mode
        self.model.eval()
        self.criterion.eval()

        for i, data in enumerate(self.test_loader):

            end = time.time()
            (input, target) = data
            # input = input.to(self.device, non_blocking=True)
            # target = target.to(self.device, non_blocking=True)
            target_var = target

            # compute output
            output = self.model(input)

            # measure model elapsed time
            eval_meters.model_time.update(time.time() - end)
            end = time.time()

            #compute loss
            loss = self.criterion(output, target_var)
            eval_meters.loss_time.update(time.time() - end)

            pred = output.data.argmax(1, keepdim=True)
            correct = pred.eq(target.data.view_as(pred)).sum()
            prec1 = 100. * correct.float() / input.size(0)

            _, prec5 = self.accuracy(output, target, topk=(1, 5))
            eval_meters.losses.update(loss.item(), input.size(0))
            eval_meters.top1.update(prec1.item(), input.size(0))
            eval_meters.top5.update(prec5.item(), input.size(0))

            #Eval batch ends
            self.logger.eval_batch_cli_log(eval_meters, i, len(self.test_loader))

        return eval_meters.top1.avg



if __name__ == "__main__":
    trainer = Trainer()

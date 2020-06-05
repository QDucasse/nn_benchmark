# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Squared-Hinged loss function and module

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Function

class squared_hinge_loss(Function):
    '''Squared hinge loss function'''
    @staticmethod
    def forward(ctx, predictions, targets):
        ctx.save_for_backward(predictions, targets)
        output = 1.-predictions.mul(targets)
        output[output.le(0.)] = 0.
        loss = torch.mean(output.mul(output))
        return loss

    @staticmethod
    def backward(ctx, grad_output):
       predictions, targets = ctx.saved_tensors
       output=1.-predictions.mul(targets)
       output[output.le(0.)]=0.
       grad_output.resize_as_(predictions).copy_(targets).mul_(-2.).mul_(output)
       grad_output.mul_(output.ne(0).float())
       grad_output.div_(predictions.numel())
       return grad_output, None

class SqrHingeLoss(nn.Module):
    '''Squared Hinge loss module'''
    def __init__(self):
        super(SqrHingeLoss, self).__init__()

    def forward(self, input, target):
        target = F.one_hot(target,input.shape[1])
        return squared_hinge_loss.apply(input, target)

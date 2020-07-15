# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

import matplotlib.pyplot as plt
import torch
import torchvision

# MNIST
dataset = torchvision.datasets.MNIST(
    root='data/',
    transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
)

x, _ = dataset[7776] # x is now a torch.Tensor
plt.imshow(x.numpy()[0], cmap='gray')
plt.show()
torchvision.utils.save_image(x, "data/img_MNIST.png")

from PIL import Image
img = Image.open("data/img_MNIST.png").convert("L")
img.save("data/img_MNIST_grayscale.png")

# CIFAR10
# dataset = torchvision.datasets.MNIST(
#     root='data/',
#     transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
# )
#
# x, _ = dataset[7776] # x is now a torch.Tensor
# plt.imshow(x.numpy()[0], cmap='gray')
# plt.show()
# torchvision.utils.save_image(x, "data/img_MNIST.png")

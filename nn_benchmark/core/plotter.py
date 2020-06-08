# -*- coding: utf-8 -*-

# Included in:
# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Plotting possibilities to visualize the data and obtain a visual inference

import torch
import matplotlib.pyplot as plt

class Plotter(object):
    def __init__(self, plot_set, model, classes):
        self.plot_set = plot_set
        self.model    = model
        self.classes  = classes

    def display_items(self, n_rows, n_columns):
        fig = plt.figure()
        for index in range(1, n_columns * n_rows + 1):
            plt.subplot(n_rows, n_columns, index)
            plt.axis('off')
            plt.imshow(self.plot_set.data[index],cmap='gray_r')

            title = 'Ground Truth: {0}'.format(self.plot_set.targets[index])

            plt.title(title, fontsize=7)
        fig.suptitle(self.model.name + ' - instances')
        plt.show()

    def display_predictions(self, n_rows, n_columns):
        fig = plt.figure()
        for index in range(1, n_columns * n_rows + 1):
            plt.subplot(n_rows, n_columns, index)
            plt.axis('off')
            plt.imshow(self.plot_set.data[index], cmap='gray_r')

            with torch.no_grad():
                self.model.eval()
                probs = self.model(self.plot_set[index][0].unsqueeze(0))

            title = f'{self.classes[torch.argmax(probs)]} ({torch.max(probs * 100):.0f}%)'

            plt.title(title, fontsize=7)
        fig.suptitle(self.model.name + ' - predictions')
        plt.show()

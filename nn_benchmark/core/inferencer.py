# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Trained model loading and inference

import torch
import torchvision
import numpy as np
import matplotlib.pyplot as plt

from nn_benchmark.networks import LeNet
from nn_benchmark.core     import Trainer # Class definition must be visible to load

class Inferencer(object):
    def __init__(self,batch_size,network):
        self.loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('data/GTSRB/test',
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize((32, 32)),
                                       torchvision.transforms.ToTensor(),
                                       torchvision.transforms.Normalize(
                                     (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])),
                batch_size=batch_size, shuffle=True, num_workers=4)

        self.classes = ['Speed limit (20km/h)',                'Speed limit (30km/h)',
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

        self.network = network
        self.load_trained_network()

    def display_6predictions(self):
        '''Displays the output of running the network over 6 inputs from the dataset'''
        # Run the network through some examples
        examples = enumerate(self.loader)
        batch_idx, (example_data, example_targets) = next(examples)
        with torch.no_grad():
            output = network(example_data)

        # Plot the predictions
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            # plt.tight_layout()
            img = example_data[i] / 2 + 0.5
            npimg = np.transpose(img.numpy(), (1,2,0))
            plt.imshow(npimg, interpolation='none')
            plt.title("Prediction: {}".format(
                self.classes[output.data.max(1, keepdim=True)[1][i].item()]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

    def load_trained_network(self):
        '''Load a saved trained network model'''
        path = 'results/trained_' + self.network.name + '.tar'
        dict = torch.load(path)
        self.network.load_state_dict(dict['network_state_dict'])
        self.network.eval()

if __name__ == "__main__":
    network = LeNet()
    inferencer = Inferencer(batch_size = 1000, network = network)
    inferencer.display_6predictions()

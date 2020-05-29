# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# Dataset exploration and visualisation

import torch
import torchvision
import numpy as np
from torch.autograd import Variable
import matplotlib.pyplot as plt

class Visualiser(object):
    def __init__(self,batch_size):
        self.loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder('data/GTSRB/train/',
                                     transform=torchvision.transforms.Compose([
                                       torchvision.transforms.Resize((32, 32)),
                                       torchvision.transforms.ToTensor(),
                                     ])),
                batch_size=batch_size, shuffle=False, num_workers=4)

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

    def display_6items(self):
        '''Shows six items along with their labels out of the dataset'''
        examples = enumerate(self.loader)
        batch_idx, (example_data, example_targets) = next(examples)
        print(example_targets)
        fig = plt.figure()
        for i in range(6):
            plt.subplot(2,3,i+1)
            # plt.tight_layout()
            img = example_data[i]
            npimg = np.transpose(img.numpy(), (1,2,0))
            plt.imshow(npimg, interpolation='none')
            plt.title("Ground Truth: {}".format(example_targets[i]))
            plt.xticks([])
            plt.yticks([])
        plt.show()

# Display 6 images of the dataset

if __name__ == "__main__":
    # Visualiser initialization and image display
    visualiser = Visualiser(batch_size = 6)
    print("All classes:")
    for id,name in enumerate(visualiser.classes):
        print("{0} -- {1}".format(id,name))
    visualiser.display_6items()

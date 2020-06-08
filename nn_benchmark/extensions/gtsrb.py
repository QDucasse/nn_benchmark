# -*- coding: utf-8 -*-

# nn_benchmark
# author - Quentin Ducasse
# https://github.com/QDucasse
# quentin.ducasse@ensta-bretagne.org

# GTSRB Dataset

import codecs
import csv
import os
import torch
import shutil
import string
import warnings
import numpy as np

from torchvision.datasets.utils import download_and_extract_archive
from torchvision.datasets import ImageFolder

from torchvision import transforms

class GTSRB(ImageFolder):
    """`GTSRB <http://benchmark.ini.rub.de/>`_ Dataset.

    Args:
        root (string): Root directory of dataset where ``GTSRB/training/``
            and  ``GTSRB/test`` exist.
        train (bool, optional): If True, creates dataset from ``GTSRB/training/``,
            otherwise from ``GTSRB/test``.
        download (bool, optional): If true, downloads the dataset from the internet and
            puts it in root directory. If dataset is already downloaded, it is not
            downloaded again.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
    """

    def __init__(self, root, train=True, transform=None, target_transform=None,
                 download=False):
        self.root = root
        self.resources = [
            ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Training_Images.zip", "f33fd80ac59bff73c82d25ab499e03a3", self.archive_train, "GTSRB_Final_Training_Images.zip"),
            ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_Images.zip",     "c7e4e6327067d32654124b0fe9e82185", self.archive_test,  "GTSRB_Final_Test_Images.zip"),
            ("https://sid.erda.dk/public/archives/daaeac0d7ce1152aea9b61d9f1e19370/GTSRB_Final_Test_GT.zip",         "fe31e9c9270bbcd7b84b7f21a9d9d9e5", self.archive_csv,   "GTSRB_Final_Test_GT.zip")
        ]

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

        if download: #download
            self.download()

        if not self._check_exists():
            raise RuntimeError('Dataset not found.' +
                               ' You can use download=True to download it')

        if train:
            super(GTSRB, self).__init__(self.root+'/GTSRB/train/', transform=transform,
                                        target_transform=target_transform)
        else:
            super(GTSRB, self).__init__(root+'/GTSRB/test', transform=transform,
                                        target_transform=target_transform)


    @property
    def train_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'train')

    @property
    def test_folder(self):
        return os.path.join(self.root, self.__class__.__name__, 'test')

    @property
    def archive_train(self):
        return os.path.join(self.root, self.__class__.__name__, 'archive_train')

    @property
    def archive_test(self):
        return os.path.join(self.root, self.__class__.__name__, 'archive_test')

    @property
    def archive_csv(self):
        return os.path.join(self.root, self.__class__.__name__, 'archive_csv')


    # @property
    # def class_to_idx(self):
    #     return {_class: i for i, _class in enumerate(self.classes)}

    def _check_exists(self):
        return (os.path.exists(self.train_folder) and
                os.path.exists(self.test_folder))

    def download(self):
        """Download the GTSRB data if it doesn't exist already."""
        # If the images are found, the download is not needed
        if self._check_exists():
            return

        # Download both training images, test images and test labels
        for url, md5, archive_folder, filename in self.resources:
            os.makedirs(archive_folder, exist_ok=True)
            download_and_extract_archive(url, download_root=archive_folder, filename=filename, md5=md5)
        # Downloading finished, processing begininng
        print("Download finished, beginning processing...")
        # MOVE images from GTSRB/Final_Training/Images/ ---> GTSRB/train
        training_classes = os.listdir(self.archive_train+'/GTSRB/Final_Training/Images/')
        for cls in training_classes:
            shutil.move(self.archive_train+'/GTSRB/Final_Training/Images/'+cls, self.train_folder)
        # REMOVE archive and unzipped files
        shutil.rmtree(self.archive_train)

        # CREATE class folders in GTSRB/test (0 -> 42)
        for cls in range(43):
            os.makedirs(self.test_folder+"/"+str(cls).zfill(5))
        # LOAD the csv giving the labels of the test images
        # MOVE each test image in the corresponding class folder
        with open(self.archive_csv+"/GT-final_test.csv") as csvfile:
            reader = csv.DictReader(csvfile,delimiter=';')
            test_images = os.listdir(self.archive_test+'/GTSRB/Final_Test/Images')
            for row in reader:
                # print(reader.fieldnames)
                filename = row['Filename']
                classid  = row['ClassId']
                for img in test_images:
                    # print("Image"+img)
                    # print("CSV:"+filename)
                    if img == filename:
                        src = self.archive_test+'/GTSRB/Final_Test/Images/'+img
                        dst = self.test_folder+"/"+str(classid).zfill(5)+"/"+img
                        shutil.move(src, dst)
        # REMOVE archive
        shutil.rmtree(self.archive_test)
        shutil.rmtree(self.archive_csv)

        print('Done!')


if __name__ == "__main__":
    train_set = GTSRB(root='data',
                      train=True,
                      download=True,
                      transform=transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.ToTensor()]))

    test_set = GTSRB(root='data',
                      train=False,
                      download=True,
                      transform=transforms.Compose([transforms.Resize((32, 32)),
                                                    transforms.ToTensor()]))

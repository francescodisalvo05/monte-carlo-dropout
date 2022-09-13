"""
Dataset link: https://www.kaggle.com/competitions/dogs-vs-cats/data?select=train.zip

Since I used only the training set, I expect to receive the filenames per split (train/val/test).
Therefore all the filenames were selected on the main pipeline and they will be manually split
in three list, each one used for creating a Dataset.
"""

from torchvision.transforms import ToTensor, Resize
from torchvision import transforms

from PIL import Image
import numpy as np

import torch
import os


class Dataset:

    def __init__(self, root_path, filenames, transform = None):
        """
        :param root_path: (str) path of the root folder containing all the available images
        :param filenames: (list) filenames involved on the current dataset split (train/val/test)
        :param labels: (list) labels involved on the current dataset split (train/val/test)
        :param transform: (transform) data transformation pipeline
        """
        self.root_path = root_path
        self.filenames = filenames
        self.transform = transform

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        """Return the couple (image,label). Let us assume that class 1 is for "dog" and class 0 is.
        There is no particular reason, I just prefer dogs :D (just kidding)
        """

        filepath = os.path.join(self.root_path,self.filenames[idx])

        # filenames are in the form "cat.0.jpg"
        # therefore let's pick the token before the first dot
        label_str = self.filenames[idx].split(".")[0]

        label = 1 if label_str == 'dog' else 0

        # read image as a numpy array
        img = np.array(Image.open(filepath))

        if self.transform:
            img = self.transform(img)

        return img,label
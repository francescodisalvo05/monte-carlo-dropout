from sklearn.model_selection import train_test_split
from data.dataset import Dataset

from torch.utils.data import DataLoader


import torch, torchvision
import torch.nn as nn
import numpy as np
import os
import random


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def get_dataloaders(root_path, batch_size, transform):

    # get only 1000 samples
    filenames = os.listdir(root_path)[:500] + os.listdir(root_path)[:-500:-1]
    labels = [1 if f.split(".")[0] == 'dog' else 0 for f in filenames]

    x_trainval, x_test, y_trainval, y_test = train_test_split(filenames, labels, stratify=labels, test_size=0.2)
    x_train, x_val, y_train, y_val = train_test_split(x_trainval, y_trainval, stratify=y_trainval, test_size=0.25)

    train_ds = Dataset(root_path, x_train, transform)
    val_ds = Dataset(root_path, x_val, transform)
    test_ds = Dataset(root_path, x_test, transform)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False)

    return train_loader, val_loader, test_loader


def get_pretrained_resnet():

    model = torchvision.models.resnet18(pretrained=True)

    # fine tune only the last fully connected layer

    # get the number of input features
    num_features = model.fc.in_features

    # update the number of input features and the number
    # of output features for our task (2: dog and cat, 1 and 0)
    model.fc = nn.Linear(in_features=num_features, out_features=2)

    return model
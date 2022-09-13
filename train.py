
from argparse import ArgumentParser


from utils.utils import set_seed, get_dataloaders

from torchvision.transforms import ToTensor, Resize
from torchvision import transforms

from model.cnn import CNN
from model.trainer import Trainer

import torch.optim as optim
import torch.nn as nn
import torch

from tqdm import tqdm

import os

def main(args):

    transform = transforms.Compose([
        ToTensor(),
        Resize((128, 128))
    ])

    train_loader, val_loader, test_loader = get_dataloaders(args.root_path, args.batch_size, transform)

    model = CNN()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(optimizer, criterion, args.output_path)


    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train(model, train_loader, epoch)
        print(f"\tTrain loss: {train_loss:.6f} \t Train accuracy: {train_acc:.2f}")

        val_loss, val_acc = trainer.validate(model, val_loader)
        print(f"\tVal loss: {val_loss:.6f} \t Val accuracy: {val_acc:.2f}")

    # save the ckpt for later
    #model.save_ckpt('...')


if __name__ == '__main__':

    parser = ArgumentParser()

    # -- model settings
    parser.add_argument('-b', '--batch_size', type=int, default=32)
    parser.add_argument('-e', '--epochs', type=int, default=10)

    # -- paths
    parser.add_argument('-R', '--root_path', type=str, default='assets/dataset/train',
                        help='Set the root path of the available images')
    parser.add_argument('-O', '--output_path', type=str, default='assets/output/',
                        help='Set the root path of the final model predictions')
    parser.add_argument('-C', '--ckpt_path', type=str, default='assets/checkpoints/',
                        help='Set the path for storing the models\' checkpoints')

    args = parser.parse_args()

    set_seed(50)

    main(args)

from argparse import ArgumentParser


from utils.utils import set_seed, get_dataloaders, get_pretrained_resnet

from torchvision.transforms import ToTensor, Resize
from torchvision import transforms

from model.trainer import Trainer

import torch.optim as optim
import torch.nn as nn
import torch
import torchvision

from model.cnn import CNN

from tqdm import tqdm

import os

def main(args):

    transform = transforms.Compose([
        ToTensor(),
        Resize((256, 256))
    ])

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    train_loader, val_loader, _ = get_dataloaders(args.root_path, args.batch_size, transform)

    # resnet18 = get_pretrained_resnet()

    model = CNN()

    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    trainer = Trainer(optimizer, criterion, args.output_path, device)


    for epoch in range(args.epochs):
        train_loss, train_acc = trainer.train(model, train_loader, epoch)
        print(f"\tTrain loss: {train_loss:.6f} \t Train accuracy: {train_acc:.2f}")

        val_loss, val_acc = trainer.validate(model, val_loader)
        print(f"\tVal loss: {val_loss:.6f} \t Val accuracy: {val_acc:.2f}")

    torch.save(model.state_dict(), os.path.join(args.ckpt_path,'ckpt.pth'))

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
from utils.utils import set_seed, get_dataloaders, get_pretrained_resnet
from argparse import ArgumentParser

from torchvision.transforms import ToTensor, Resize
from torchvision import transforms

import torchvision
import torch

import numpy as np


def main(args):

    # load model checkpoint
    model = get_pretrained_resnet()
    model.load_state_dict(torch.load(args.model_path))
    model.eval()

    # transform
    transform = transforms.Compose([
        ToTensor(),
        Resize((128, 128))
    ])

    # set device
    device = 'cuda:1' if torch.cuda.is_available() else 'cpu'

    # get data
    _, _, test_loader = get_dataloaders(args.root_path, 1, transform)

    labels, predictions = [], []

    for idx, (image,label) in enumerate(test_loader):

        image, label = image.to(device), label.to(device)

        output = model(image)
        output_max_idx = np.argmax(output.cpu().detach().numpy())

        predictions.append(output_max_idx)
        labels.append(label.cpu().detach().numpy())

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)



    print(f'Test accuracy {len(labels == predictions)}/{len(predictions)} : {len(labels == predictions)/len(predictions):.2f}')






if __name__ == '__main__':

    parser = ArgumentParser()

    # -- paths
    parser.add_argument('-R', '--root_path', type=str, default='assets/dataset/train',
                        help='Set the root path of the available images')
    parser.add_argument('-O', '--output_path', type=str, default='assets/predictions/',
                        help='Set the root path of the predictions')
    parser.add_argument('-M', '--model_path', type=str, default='assets/checkpoints/ckpt.pth',
                        help='Set the checkpoint filepath')


    args = parser.parse_args()

    set_seed(50)

    main(args)
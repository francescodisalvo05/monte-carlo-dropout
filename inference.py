from utils.utils import set_seed, get_dataloaders
from argparse import ArgumentParser

from torchvision.transforms import ToTensor, Resize
from torchvision import transforms

from model.cnn import CNN

from tqdm import tqdm

import torchvision
import torch

import numpy as np
import os


def main(args):

    # set device
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

    # load model checkpoint
    model = CNN()
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # transform
    transform = transforms.Compose([
        ToTensor(),
        Resize((256, 256))
    ])



    # get data
    _, _, test_loader = get_dataloaders(args.root_path, 1, transform)

    labels, predictions, pos_probabilities = [], [], []

    for idx, (image,label) in tqdm(enumerate(test_loader)):

        image, label = image.to(device), label.to(device)

        output = model(image, train=False)
        output_max_idx = np.argmax(output.cpu().detach().numpy()[0])

        predictions.append(output_max_idx)
        labels.append(label.cpu().detach().numpy()[0])


        pos_probabilities.append(output.cpu().detach().numpy().squeeze()[1])

    labels = np.asarray(labels)
    predictions = np.asarray(predictions)
    pos_probabilities = np.asarray(pos_probabilities)

    print(f'Test accuracy {np.sum(labels == predictions)}/{len(predictions)} : {np.sum(labels == predictions)/len(predictions):.2f}')

    with open(os.path.join(args.output_path,'output.txt'), 'w') as out_file:
        # they have the same prediction order
        filenames = test_loader.dataset.filenames

        for f, p, l in zip(filenames, pos_probabilities, labels):
            line = f'{f},{p},{l}\n'
            out_file.write(line)

if __name__ == '__main__':

    parser = ArgumentParser()

    # -- paths
    parser.add_argument('-R', '--root_path', type=str, default='assets/dataset/train',
                        help='Set the root path of the available images')
    parser.add_argument('-O', '--output_path', type=str, default='assets/predictions/dropoutOFF/',
                        help='Set the root path of the predictions')
    parser.add_argument('-M', '--model_path', type=str, default='assets/checkpoints/ckpt.pth',
                        help='Set the checkpoint filepath')


    args = parser.parse_args()

    set_seed(50)

    main(args)
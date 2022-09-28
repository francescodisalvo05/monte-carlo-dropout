from utils.monte_carlo import monte_carlo_inferences, \
                              store_monte_carlo_statistics, \
                              get_average_accuracy, \
                              get_var_histplot, \
                              plot_examples
from utils.utils import set_seed, get_dataloaders
from collections import defaultdict
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

    # get model
    model = CNN()

    # transform
    transform = transforms.Compose([
        ToTensor(),
        Resize((256, 256))
    ])

    # get data
    _, _, test_loader = get_dataloaders(args.root_path, 1, transform)

    # -- MAKE INFERENCE AND STORE PREDICTIONS ON DISK -- #
    monte_carlo_inferences(args, test_loader, model, device)

    # -- EXTRACT STATISTICS (MEAN/VAR) ACROSS MONTE CARLO SAMPLES -- #
    store_monte_carlo_statistics(args)

    # -- GET AVERAGE ACCURACY -- #
    get_average_accuracy(args)

    # -- PLOT Y-CASTED HISTPLOT -- #
    get_var_histplot(args)

    # -- GET EXAMPLE OF CERTAIN / UNCERTAIN MASSES -- #
    plot_examples(args)




if __name__ == '__main__':

    parser = ArgumentParser()

    parser.add_argument('-n', '--num_inferences', type=int, default=50)

    # -- paths
    parser.add_argument('-R', '--root_path', type=str, default='assets/dataset/train',
                        help='Set the root path of the available images')
    parser.add_argument('-O', '--output_path', type=str, default='assets/predictions/dropoutON/',
                        help='Set the root path of the predictions')
    parser.add_argument('-M', '--model_path', type=str, default='assets/checkpoints/ckpt_dropout.pth',
                        help='Set the checkpoint filepath')
    parser.add_argument('-U', '--uncertainty_path', type=str, default='assets/uncertainty/',
                        help='Set the root path of the uncertainty results')

    args = parser.parse_args()

    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    if not os.path.exists(args.uncertainty_path):
        os.makedirs(args.uncertainty_path)

    set_seed(50)

    main(args)
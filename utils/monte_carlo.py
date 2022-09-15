from collections import defaultdict

import matplotlib.pyplot as plt
import seaborn as sns

import numpy as np
import torch
import os


def monte_carlo_inferences(args, test_loader, model, device):

    # load model state dict
    model.load_state_dict(torch.load(args.model_path, map_location=device))

    # enable dropout at test time
    def enable_dropout(m):
        for each_module in m.modules():
            if each_module.__class__.__name__.startswith('Dropout'):
                each_module.train()

    model.eval()
    enable_dropout(model)

    # --- INFERENCE --- #
    for inference_idx in range(1,args.num_inferences + 1):

        labels, predictions, pos_probabilities = [], [], []

        for idx, (image,label) in enumerate(test_loader):

            image, label = image.to(device), label.to(device)

            output = model(image, train=False)
            output_max_idx = np.argmax(output.cpu().detach().numpy()[0])

            predictions.append(output_max_idx)
            labels.append(label.cpu().detach().numpy()[0])
            pos_probabilities.append(output.cpu().detach().numpy().squeeze()[1])

        labels = np.asarray(labels)
        predictions = np.asarray(predictions)
        pos_probabilities = np.asarray(pos_probabilities)

        print(f'#{str(inference_idx).zfill(2)} Test accuracy {np.sum(labels == predictions)}/{len(predictions)} : {np.sum(labels == predictions)/len(predictions):.2f}')

        # --- WRITE OUTCOMES ON DISK --- #
        with open(os.path.join(args.output_path,f'output_{str(inference_idx).zfill(2)}.txt'), 'w') as out_file:
            # they have the same prediction order
            filenames = test_loader.dataset.filenames
            for f, p, l in zip(filenames, pos_probabilities, labels):
                line = f'{f},{p},{l}\n'
                out_file.write(line)


def store_monte_carlo_statistics(args):

    monte_carlo_files = os.listdir(args.output_path)

    # store them in the following way:
    # { 'cat.1.jpg : [prob1, prob2, ..., prob50}
    filename_probabilities = defaultdict(list)
    labels = []  # they will have the same order

    # remove all unnecessary files
    monte_carlo_files = [f for f in monte_carlo_files if f.split(".")[-1] == "txt"]

    # get all "dog" probabilities across each monte carlo samples for every test mass
    for idx, monte_carlo_file in enumerate(monte_carlo_files):
        with open(os.path.join(args.output_path, monte_carlo_file), 'r') as curr_file:
            for line in curr_file.readlines():
                f, p, l = line.strip().split(",")
                filename_probabilities[f].append(float(p))

                if idx == 0:
                    labels.append(int(l))

    # get average and variance probabilities
    average_probabilities, variance_probabilities = [], []
    filenames = list(filename_probabilities.keys())
    for idx, filename in enumerate(filenames):
        average_probabilities.append(np.mean(filename_probabilities[filename]))
        variance_probabilities.append(np.var(filename_probabilities[filename]))

    # store filename, average, variance and label -> they will be leveraged for studying the
    # correlation between uncertainty and performances
    with open(os.path.join(args.uncertainty_path, 'monte_carlo_statistics.txt'), 'w') as curr_file:

        for filename, average, variance, label in zip(filenames,
                                                      average_probabilities,
                                                      variance_probabilities,
                                                      labels):
            line = f'{filename},{average},{variance},{label}\n'
            curr_file.write(line)


def get_average_accuracy(args):

    correct = [] # add 1 if they match, 0 otherwise

    with open(os.path.join(args.uncertainty_path, 'monte_carlo_statistics.txt'), 'r') as curr_file:
        for line in curr_file.readlines():
            f, a, _, l = line.strip().split(",")

            if int(round(float(a))) == int(l):
                correct.append(1)
            else:
                correct.append(0)

    print(f'Accuracy on average MC samples : {np.sum(correct)/len(correct):.2f} ')


def get_std_histplot(args):

    variances = []

    with open(os.path.join(args.uncertainty_path, 'monte_carlo_statistics.txt'), 'r') as curr_file:
        for line in curr_file.readlines():
            _, _, v, _ = line.strip().split(",")
            variances.append(float(v))

    sns.displot(x=variances)
    plt.ylim((0,200))

    plt.savefig(os.path.join(args.uncertainty_path,'hist.png'), bbox_inch='tight')

def plot_examples(args):

    filenames, variances = [], []

    with open(os.path.join(args.uncertainty_path, 'monte_carlo_statistics.txt'), 'r') as curr_file:
        for line in curr_file.readlines():
            f, _, v, _ = line.strip().split(",")
            variances.append(float(v))
            filenames.append(f)

    filenames = np.asarray(filenames)
    variances = np.asarray(variances)

    ordered_variances = np.argsort(variances) # increasing order
    top_16_uncertain = filenames[ordered_variances][:-17:-1]
    top_16_certain = filenames[ordered_variances][:16]

    print(f'Top 16 certain : {top_16_certain}')
    print(f'Top 16 uncertain : {top_16_uncertain}')

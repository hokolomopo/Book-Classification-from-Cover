from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from stats_from_txt import *

if __name__ == "__main__":
    stats = get_stats("lr/lr_results_plt.txt")

    lrs = []
    accs = []
    epochs = []

    for stat in stats:
        lrs.append(stat.cnn_parameters.lr)
        accs.append([stat.accuracies['train'], stat.accuracies['val']])
        epochs.append(stat.epochs['train'])

    dummy = range(len(lrs))
    colors = ['r', 'g', 'b']

    plt.figure()
    for i in range(len(accs)):
        plt.plot(epochs[i], accs[i][0], colors[i], label="train lr {}".format(lrs[i]))
        plt.plot(epochs[i], accs[i][1], colors[i] + '--', label="val lr {}".format(lrs[i]))

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()




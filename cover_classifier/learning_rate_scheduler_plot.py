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
    stats = get_stats("results_schedulers.txt")

    schedulers = ["triangle2", "triangle"]
    accs = []
    epochs = []

    for stat in stats:
        accs.append([stat.accuracies['train'], stat.accuracies['val']])
        epochs.append(stat.epochs['train'])

    colors = ['r', 'g', 'b']

    plt.figure()
    for i in range(len(accs)):
        plt.plot(epochs[i], accs[i][0], colors[i], label="train {}".format(schedulers[i]))
        plt.plot(epochs[i], accs[i][1], colors[i] + '--', label="val {}".format(schedulers[i]))

    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()




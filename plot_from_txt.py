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
    stats = get_stats("plot.txt")

    epochs = []
    accsTrain = []
    accsVal = []

    for stat in stats:
        accsTrain.append(stat.accuracies['train'])
        accsVal.append(stat.accuracies['val'])
        epochs.append(stat.epochs["train"])

    plt.figure()

    for i in range(len(accsTrain)):
        plt.plot(epochs[i], accsTrain[i])
        plt.plot(epochs[i], accsVal[i])
    plt.show()
    plt.savefig("Adam.pdf")





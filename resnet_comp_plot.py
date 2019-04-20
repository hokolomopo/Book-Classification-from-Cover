from __future__ import print_function, division

"""
Plot a comparison time/accuracy between ResNet models
"""

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
    stats = get_stats("resnet_comparison.txt")

    times = []
    resnets = []
    accs = []

    for stat in stats:
        resnets.append(stat.cnn_parameters.resnet)
        accs.append(stat.best_acc)
        times.append(stat.time_elapsed / len(stat.epochs))

    # plt.figure()
    # dummy = range(len(resnets))
    # plt.plot(dummy, times)
    # plt.xticks(dummy, resnets)
    # plt.show()
    # plt.savefig("resnet_time_comparison.pdf")


    dummy = range(len(resnets))

    fig, ax1 = plt.subplots()
    ax1.plot(dummy, times, 'b')
    ax1.set_xlabel('Resnet Version')
    # Make the y-axis label, ticks and tick labels match the line color.
    ax1.set_ylabel('Time / Epoch (s)', color='b')
    ax1.tick_params('y', colors='b')

    ax2 = ax1.twinx()
    ax2.plot(dummy, accs, 'r')
    ax2.set_ylabel('Accuracy', color='r')
    ax2.tick_params('y', colors='r')

    plt.xticks(dummy, resnets)
    fig.tight_layout()
    plt.show()




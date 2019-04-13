"""
Transfer Learning Tutorial
==========================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_
In this tutorial, you will learn how to train your network using
transfer learning. You can read more about the transfer learning at `cs231n
notes <https://cs231n.github.io/transfer-learning/>`__
Quoting these notes,
    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.
These two major transfer learning scenarios look as follows:
-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.
"""
# License: BSD
# Author: Sasank Chilamkurthy

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




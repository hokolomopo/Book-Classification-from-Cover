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

from bookDataset import *

if __name__ == "__main__":
    trainCsvPath = "dataset/book30-listing-train.csv"
    testCsvPath = "dataset/book30-listing-test.csv"
    coverPath = "dataset/covers"

    #Create dataset
    dataset = BookDataset(trainCsvPath, coverPath)
    df3 = dataset.dataset
    df3 = df3.reset_index().drop_duplicates(subset='class', keep='last').set_index('index')
    df3 = df3.sort_values(by=['class'])
    names = df3['class_name'].tolist()

    print(df3)
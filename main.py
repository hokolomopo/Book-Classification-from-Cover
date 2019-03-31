from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from bookDataset import BookDataset

if __name__ == "__main__":
    trainCsvPath = "dataset/book30-listing-train.csv"
    testCsvPath = "dataset/book30-listing-test.csv"
    coverPath = "dataset/covers"

    #Create dataset
    dataset = BookDataset(trainCsvPath, coverPath)

    #testing stuff
    print(dataset.dataset.head(3))

    item = dataset.__getitem__(0)
    imgplot = plt.imshow(item["cover"])
    plt.savefig("test.png")
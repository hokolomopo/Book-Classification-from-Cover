from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

class BookDataset(Dataset):
    """Book dataset."""

    def __init__(self, csv_file, image_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            image_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        cols = ["index", "filename", "url", "title", "author", "class", "class_name"]
        self.dataset = pd.read_csv(csv_file, header = None, names = cols, encoding = "ISO-8859-1")

        self.image_dir = image_dir
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = self.image_dir + '/' +  self.dataset.iloc[idx, 1]
        cover = io.imread(img_name)
        line = self.dataset.iloc[idx]
        title = line["title"]
        id = line["class"]
        sample = {'cover': cover, 'title' : title, 'class' : id}

        if self.transform:
            sample = self.transform(sample)

        return sample

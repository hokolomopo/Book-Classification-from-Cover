from __future__ import print_function, division
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
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

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        cover, title, label = sample['cover'], sample['title'], sample['class']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        cover = cover.transpose((2, 0, 1))
        return {'cover': torch.from_numpy(cover).float(), 'title': title, 
                'class': label}

def create_data_loaders(train_csv_file, test_csv_file, image_dir, train_prop, 
                       batch_size, num_workers = 1):
    
    transform = ToTensor()

    train_set = BookDataset(train_csv_file, image_dir, transform)
    test_set = BookDataset(test_csv_file, image_dir, transform)
    
    """
    dataset_length = len(dataset)
    
    train_size = int(train_prop * dataset_length)
    test_size = dataset_length - train_size
    train_set, test_set = torch.utils.data.random_split(dataset, [train_size, 
                                                                  test_size])
    """
    
    data_loaders = {
        "train": DataLoader(train_set, batch_size = batch_size, shuffle = True,
                            num_workers = num_workers, pin_memory = True),
        "test": DataLoader(test_set, batch_size = batch_size, shuffle = True, 
                           num_workers = num_workers, pin_memory = True)
    }

    return data_loaders

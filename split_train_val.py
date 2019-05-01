"""
Split train set into train and validation set
"""

import pandas as pd
from sklearn.model_selection import train_test_split

dataset_path = "dataset/book30-listing-train.csv"
cols = ["index", "filename", "url", "title", "author", "class", "class_name"]
dataset = pd.read_csv(dataset_path, header = None, names = cols, encoding = "ISO-8859-1")

train_set, validation_set = train_test_split(dataset, test_size = 0.2)

train_set.to_csv("dataset/train_set.csv", index = False, encoding = "ISO-8859-1", header = False)
validation_set.to_csv("dataset/validation_set.csv", index = False, encoding = "ISO-8859-1", header = False)
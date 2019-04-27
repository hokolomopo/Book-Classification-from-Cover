from sentence_encoder import SentenceEmbedding, SentenceEmbeddingGlove
import nltk
from InferSent.models import InferSent
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import pickle
from torchvision import transforms
from PIL import Image

class CombinedBookDataset(Dataset):
    """Book dataset."""

    def __init__(self, csv_file, datasetTransform, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """

        cols = ["index", "filename", "url", "title", "author", "class", "class_name"]
        self.dataset = pd.read_csv(csv_file, header = None, names = cols, encoding = "ISO-8859-1")
        self.titles = datasetTransform.transform_titles(self.dataset)

        self.transform = transform

        #Create list of classes
        df = self.dataset.reset_index().drop_duplicates(subset='class', keep='last').set_index('index')
        df = df.sort_values(by=['class'])
        self.classes = df['class_name'].tolist()

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        img_name = "dataset/covers/" +  self.dataset.iloc[idx, 1]
        cover = Image.open(img_name)
        line = self.dataset.iloc[idx]
        title = self.titles[idx]
        label = line["class"]

        if self.transform:
            cover = self.transform(cover)

        return ((cover, torch.from_numpy(title).float()), label)

def create_combined_data_loaders(train_csv_file, val_csv_file, test_csv_file, batch_size, num_workers = 1, word_emb = "FastText"):
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    if word_emb == "FastText":
        datasetTransform = SentenceEmbedding([train_csv_file, val_csv_file, test_csv_file])
    elif word_emb == "Glove":
        datasetTransform = SentenceEmbeddingGlove([train_csv_file, val_csv_file, test_csv_file])
    else:
        return None

    print("creating datasets")
    train_set = CombinedBookDataset(train_csv_file, datasetTransform = datasetTransform, transform = data_transforms["train"])
    val_set = CombinedBookDataset(val_csv_file, datasetTransform = datasetTransform, transform = data_transforms["val"])
    test_set = CombinedBookDataset(test_csv_file, datasetTransform = datasetTransform, transform = data_transforms["test"])

    print("creating dataloaders")
    data_loaders = {
        "train": DataLoader(train_set, batch_size = batch_size, shuffle = True,
                            num_workers = num_workers),
        "val": DataLoader(val_set, batch_size = batch_size, shuffle = True, 
                          num_workers = num_workers),
        "test": DataLoader(test_set, batch_size = batch_size, shuffle = True, 
                           num_workers = num_workers)
    }

    return data_loaders

def save_combined_data_loaders(pickle_file_name, batch_size, num_workers = 0, word_emb = "FastText"):
    train_csv_path = "dataset/train_set_cleaned.csv"
    val_csv_path = "dataset/validation_set_cleaned.csv"
    test_csv_path = "dataset/book30-listing-test_cleaned.csv"

    data_loaders = create_combined_data_loaders(train_csv_path, val_csv_path, test_csv_path, batch_size, num_workers, word_emb)
    if data_loaders:
        print("pickling dataloaders")
        with open(pickle_file_name, "wb") as fp:
            pickle.dump(data_loaders, fp)

    else:
        print("Invalid word_emb arg")

def save_final_combined_data_loaders(pickle_file_name, batch_size, num_workers = 0, word_emb = "FastText"):
    train_csv_path = "dataset/book30-listing-train_cleaned.csv"
    val_csv_path = "dataset/book30-listing-test_cleaned.csv"
    test_csv_path = "dataset/book30-listing-test_cleaned.csv"

    data_loaders = create_combined_data_loaders(train_csv_path, val_csv_path, test_csv_path, batch_size, num_workers, word_emb)
    if data_loaders:
        print("pickling dataloaders")
        with open(pickle_file_name, "wb") as fp:
            pickle.dump(data_loaders, fp)

    else:
        print("Invalid word_emb arg")

def save_combined_10_classes_data_loaders(pickle_file_name, batch_size, num_workers = 0, word_emb = "FastText"):
    train_csv_path = "dataset/train_set_cleaned_10.csv"
    val_csv_path = "dataset/validation_set_cleaned_10.csv"
    test_csv_path = "dataset/book30-listing-test_cleaned_10.csv"

    data_loaders = create_combined_data_loaders(train_csv_path, val_csv_path, test_csv_path, batch_size, num_workers, word_emb)
    if data_loaders:
        print("pickling dataloaders")
        with open(pickle_file_name, "wb") as fp:
            pickle.dump(data_loaders, fp)

    else:
        print("Invalid word_emb arg")

if __name__ == "__main__":
    BATCH_SIZES = [4, 8, 16, 32, 64]
    nltk.download('punkt')
    
    """
    for batch_size in BATCH_SIZES:
        pickle_file_name = "dataloaders/combined_data_loaders_{}.pickle".format(batch_size)
        save_combined_data_loaders(pickle_file_name, batch_size, 0)
        pickle_file_name = "dataloaders/final_combined_data_loaders_{}.pickle".format(batch_size)
        save_final_combined_data_loaders(pickle_file_name, batch_size, 0)
    """
    save_combined_10_classes_data_loaders("dataloaders/combined_data_loaders_32_10.pickle", 32)
import re
import torch
from torchtext import data
from torchtext.vocab import Vectors, GloVe
import pandas as pd
from torchvision import transforms
from PIL import Image
from cnn_text_data import RawTextBookDataset

class CombinedRawTextBookDataset(data.Dataset):
	"""
	Inspired from
	https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/mydatasets.py#L89
	"""	

	def __init__(self, title_field, cover_field, label_field, csv_file, coverTransform = None):

		def clean_str(string):
			"""
			Tokenization/string cleaning for all datasets except for SST.
			Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
			"""
			string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
			string = re.sub(r"\'s", " \'s", string)
			string = re.sub(r"\'ve", " \'ve", string)
			string = re.sub(r"n\'t", " n\'t", string)
			string = re.sub(r"\'re", " \'re", string)
			string = re.sub(r"\'d", " \'d", string)
			string = re.sub(r"\'ll", " \'ll", string)
			string = re.sub(r",", " , ", string)
			string = re.sub(r"!", " ! ", string)
			string = re.sub(r"\(", " \( ", string)
			string = re.sub(r"\)", " \) ", string)
			string = re.sub(r"\?", " \? ", string)
			string = re.sub(r"\s{2,}", " ", string)
			return string.strip()

		title_field.preprocessing = data.Pipeline(clean_str)

		def cover_preprocessing(img_name):
			cover = Image.open(img_name)
			print("after open: ".format(cover))
			print(type(cover))
			if coverTransform:
				cover = coverTransform(cover)
			print("after transform".format(cover))
			print(type(cover))

			return cover

		cover_field.preprocessing = data.Pipeline(cover_preprocessing)

		fields = [("title", title_field), ("cover", cover_field), ("label", label_field)]

		examples = []
		cols = ["index", "filename", "url", "title", "author", "class", "class_name"]
		dataset = pd.read_csv(csv_file, header = None, names = cols, encoding = "ISO-8859-1")
		for i, row in dataset.iterrows():
			print(i)
			img_name = "dataset/covers/" + row["filename"]
			print(img_name)
			examples.append(data.Example.fromlist([row["title"], img_name, row["class"]], fields))

		super().__init__(examples, fields)

def create_combined_text_iterators(train_csv_file, val_csv_file, test_csv_file, batch_size, num_workers = 0, validation = True):
	MAX_LENGTH = 50

	def tokenize(title):
		if len(title) > MAX_LENGTH:
			title = title[:MAX_LENGTH]

		return title.split()

	TITLE = data.Field(sequential = True, tokenize = tokenize, lower = True, include_lengths = False, batch_first = True, fix_length = MAX_LENGTH)
	IMAGE_TRAIN = data.RawField()
	IMAGE_TRAIN.is_target = False
	IMAGE_VAL = data.RawField()
	IMAGE_VAL.is_target = False
	IMAGE_TEST = data.RawField()
	IMAGE_TEST.is_target = False
	LABEL = data.Field(sequential = False, is_target = True)

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

	print("creating datasets")
	train_set = CombinedRawTextBookDataset(TITLE, IMAGE_TRAIN, LABEL, train_csv_file, data_transforms["train"])

	if validation:
		val_set = CombinedRawTextBookDataset(TITLE, IMAGE_VAL, LABEL, val_csv_file, data_transforms["val"])
		test_set = RawTextBookDataset(TITLE, LABEL, test_csv_file)
	else:
		val_set = RawTextBookDataset(TITLE, LABEL, val_csv_file)
		test_set = CombinedRawTextBookDataset(TITLE, IMAGE_VAL, LABEL, val_csv_file, data_transforms["test"])

	TITLE.build_vocab(train_set, val_set, test_set, vectors = GloVe(name='6B', dim=300))
	LABEL.build_vocab(train_set, val_set, test_set)

	print("creating dataloaders")

	if validation:
		iterators = {
			"train": data.Iterator(train_set, batch_size, shuffle = True),
			"val": data.Iterator(validation_set, batch_size, shuffle = True)
		}
	else:
		iterators = {
			"train": data.Iterator(train_set, batch_size, shuffle = True),
			"test": data.Iterator(test_set, batch_size, shuffle = True)
		}


	word_embedding = TITLE.vocab.vectors

	return TITLE, word_embedding, iterators
import re
import torch
from torchtext import data
from torchtext.vocab import Vectors, GloVe
import pandas as pd

class RawTextBookDataset(data.Dataset):
	"""
	Inspired from
	https://github.com/srviest/char-cnn-text-classification-pytorch/blob/master/mydatasets.py#L89
	"""	

	def __init__(self, title_field, label_field, csv_file):

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
		fields = [("title", title_field), ("label", label_field)]

		examples = []
		cols = ["index", "filename", "url", "title", "author", "class", "class_name"]
		dataset = pd.read_csv(csv_file, header = None, names = cols, encoding = "ISO-8859-1")
		for i, row in dataset.iterrows():
			examples.append(data.Example.fromlist([row["title"], row["class"]], fields))

		super().__init__(examples, fields)

def create_raw_text_iterators(train_csv_file, val_csv_file, test_csv_file, batch_size, num_workers = 0):
	"""
	Create iterators for text classification with concolutionnal networks
	"""
	
	MAX_LENGTH = 50

	def tokenize(title):
		if len(title) > MAX_LENGTH:
			title = title[:MAX_LENGTH]

		return title.split()

	TITLE = data.Field(sequential = True, tokenize = tokenize, lower = True, include_lengths = False, batch_first = True, fix_length = MAX_LENGTH)
	LABEL = data.Field(sequential = False, is_target = True)

	print("creating datasets")
	train_set = RawTextBookDataset(TITLE, LABEL, train_csv_file)
	val_set = RawTextBookDataset(TITLE, LABEL, val_csv_file)
	test_set = RawTextBookDataset(TITLE, LABEL, test_csv_file)

	TITLE.build_vocab(train_set, val_set, test_set, vectors = GloVe(name='6B', dim=300))
	LABEL.build_vocab(train_set, val_set, test_set)

	print("creating dataloaders")
	iterators = {
		"train": data.Iterator(train_set, batch_size, shuffle = True),
		"val": data.Iterator(val_set, batch_size, shuffle = True),
		"test": data.Iterator(test_set, batch_size, shuffle = True)
	}

	word_embedding = TITLE.vocab.vectors

	return TITLE, word_embedding, iterators
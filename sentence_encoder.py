import nltk
from InferSent.models import InferSent
from torch.utils.data import Dataset, DataLoader
import torch
import pandas as pd
import pickle

class TextBookDataset(Dataset):
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
		line = self.dataset.iloc[idx]
		title = self.titles[idx]
		label = line["class"]

		if self.transform:
			cover = self.transform(cover)

		return (torch.from_numpy(title).float(), label)

class SentenceEmbedding():

	def __init__(self, csvFiles):
		"""
		csvFiles: A list of csv files containing the datasets used.
		"""
		V = 2
		MODEL_PATH = 'InferSent/encoder/infersent%s.pickle' % V
		params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
				'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

		self.infersent = InferSent(params_model)
		self.infersent.cuda()
		self.infersent.load_state_dict(torch.load(MODEL_PATH))

		W2V_PATH = 'InferSent/dataset/fastText/crawl-300d-2M-subword.vec'
		self.infersent.set_w2v_path(W2V_PATH)

		sentences = []
		for file in csvFiles:
			cols = ["index", "filename", "url", "title", "author", "class", "class_name"]
			dataset = pd.read_csv(file, header = None, names = cols, encoding = "ISO-8859-1")
			titles = dataset['title'].tolist()

			sentences.extend(titles)

		self.infersent.build_vocab(sentences, tokenize=True)

	def transform_titles(self, dataset):
		transformedTitles = self.infersent.encode(dataset['title'], tokenize=True)
		print(transformedTitles.shape)
		print(type(transformedTitles))
		print(type(transformedTitles[0]))
		return transformedTitles

class SentenceEmbeddingGlove():

	def __init__(self, csvFiles):
		"""
		csvFiles: A list of csv files containing the datasets used.
		"""
		V = 1
		MODEL_PATH = 'InferSent/encoder/infersent%s.pickle' % V
		params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
				'pool_type': 'max', 'dpout_model': 0.0, 'version': V}

		self.infersent = InferSent(params_model)
		self.infersent.cuda()
		self.infersent.load_state_dict(torch.load(MODEL_PATH))

		W2V_PATH = 'InferSent/dataset/fastText/crawl-300d-2M-subword.vec'
		self.infersent.set_w2v_path(W2V_PATH)

		sentences = []
		for file in csvFiles:
			cols = ["index", "filename", "url", "title", "author", "class", "class_name"]
			dataset = pd.read_csv(file, header = None, names = cols, encoding = "ISO-8859-1")
			titles = dataset['title'].tolist()

			sentences.extend(titles)

		self.infersent.build_vocab(sentences, tokenize=True)

	def transform_titles(self, dataset):
		transformedTitles = self.infersent.encode(dataset['title'], tokenize=True)
		print(transformedTitles.shape)
		print(type(transformedTitles))
		print(type(transformedTitles[0]))
		return transformedTitles

def create_text_data_loaders(train_csv_file, val_csv_file, test_csv_file, batch_size, num_workers = 1, word_emb = "FastText"):
	if word_emb == "FastText":
		datasetTransform = SentenceEmbedding([train_csv_file, val_csv_file, test_csv_file])
	elif word_emb == "Glove":
		datasetTransform = SentenceEmbeddingGlove([train_csv_file, val_csv_file, test_csv_file])
	else:
		return None

	print("creating datasets")
	train_set = TextBookDataset(train_csv_file, datasetTransform = datasetTransform)
	val_set = TextBookDataset(val_csv_file, datasetTransform = datasetTransform)
	test_set = TextBookDataset(test_csv_file, datasetTransform = datasetTransform)

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

def save_text_data_loaders(pickle_file_name, batch_size, num_workers = 0, word_emb = "FastText"):
	train_csv_path = "dataset/train_set_cleaned.csv"
	val_csv_path = "dataset/validation_set_cleaned.csv"
	test_csv_path = "dataset/book30-listing-test_cleaned.csv"

	data_loaders = create_text_data_loaders(train_csv_path, val_csv_path, test_csv_path, batch_size, num_workers, word_emb)
	if data_loaders:
		print("pickling dataloaders")
		with open(pickle_file_name, "wb") as fp:
			pickle.dump(data_loaders, fp)

	else:
		print("Invalid word_emb arg")

def save_final_text_data_loaders(pickle_file_name, batch_size, num_workers = 0, word_emb = "FastText"):
	train_csv_path = "dataset/book30-listing-train_cleaned.csv"
	val_csv_path = "dataset/book30-listing-test_cleaned.csv"
	test_csv_path = "dataset/book30-listing-test_cleaned.csv"

	data_loaders = create_text_data_loaders(train_csv_path, val_csv_path, test_csv_path, batch_size, num_workers, word_emb)
	if data_loaders:
		print("pickling dataloaders")
		with open(pickle_file_name, "wb") as fp:
			pickle.dump(data_loaders, fp)

	else:
		print("Invalid word_emb arg")

if __name__ == "__main__":
	BATCH_SIZES = [4, 8, 16, 32, 64]
	nltk.download('punkt')
	
	for batch_size in BATCH_SIZES:
		pickle_file_name = "dataloaders/encoded_text_data_loaders_{}.pickle".format(batch_size)
		save_text_data_loaders(pickle_file_name, batch_size, 0)
		pickle_file_name = "dataloaders/final_encoded_text_data_loaders_{}.pickle".format(batch_size)
		save_final_text_data_loaders(pickle_file_name, batch_size, 0)

		pickle_file_name = "dataloaders/encoded_text_data_loaders_glove_{}.pickle".format(batch_size)
		save_text_data_loaders(pickle_file_name, batch_size, 0, "Glove")
		pickle_file_name = "dataloaders/final_encoded_text_data_loaders_glove_{}.pickle".format(batch_size)
		save_final_text_data_loaders(pickle_file_name, batch_size, 0, "Glove")

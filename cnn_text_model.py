import torch
import torch.nn as nn
from torch.nn import functional as F
from cnn_text_data import create_raw_text_iterators
from train_text_iterators import train_model
import torch.optim as optim

class CnnTitleClassifier(nn.Module):
	"""
	Inspired from:
	Kim, Y., Convolutional Neural Networks for Sentence Classification. Proceedings of the 2014
	Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 2014.
	"""
	def __init__(self, vocab_size, embedding_length, embedding_weights):
		super().__init__()

		self.embed = nn.Embedding(vocab_size, embedding_length)
		self.embed.weights = nn.Parameter(embedding_weights, requires_grad=False)

		self.out_channels = 100

		self.conv1 = nn.Conv2d(1, self.out_channels, (3, embedding_length))
		self.conv2 = nn.Conv2d(1, self.out_channels, (4, embedding_length))
		self.conv3 = nn.Conv2d(1, self.out_channels, (5, embedding_length))
		self.dropout = nn.Dropout(0.5)
		self.relu = nn.ReLU()
		self.fc = nn.Sequential(
			nn.Linear(3 * self.out_channels, 30),
			nn.Softmax(0)
			)

	def conv(self, input, conv_layer):
		conv_out = conv_layer(input)

		# Remove dimension of size 1 corresponding to the embedding
		conv_out = conv_out.squeeze(3)
		activation_out = self.relu(conv_out)

		# Max over word dimension, kernel size = word dimension
		pool_out = F.max_pool1d(activation_out, activation_out.size()[2])

		# Remove word dimension
		pool_out = pool_out.squeeze(2)

		# Dimensions = batch_size, out_channels
		return pool_out


	def forward(self, titles):
		emb_titles = self.embed(titles)
		conv_input = emb_titles.unsqueeze(1)

		conv_out1 = self.conv(conv_input, self.conv1)
		conv_out2 = self.conv(conv_input, self.conv1)
		conv_out3 = self.conv(conv_input, self.conv1)

		# concat channels
		conv_out = torch.cat((conv_out1, conv_out2, conv_out3), 1)
		conv_out = self.dropout(conv_out)

		proba = self.fc(conv_out)

		return proba

def create_model_iterators(train_csv_file, val_csv_file, test_csv_file, batch_size):
	EMBEDDING_LENGTH = 300

	TITLE, word_embedding, iterators = create_raw_text_iterators(train_csv_file, val_csv_file, test_csv_file, batch_size)
	model = CnnTitleClassifier(len(TITLE.vocab), EMBEDDING_LENGTH, word_embedding)

	return model, iterators

def test_model():
	TRAIN_CSV_FILE = "dataset/train_set_cleaned.csv"
	VAL_CSV_FILE = "dataset/validation_set_cleaned.csv"
	TEST_CSV_FILE = "dataset/book30-listing-test_cleaned.csv"

	BATCH_SIZE = 32

	EPOCHS = 200

	LR = 0.001

	model, iterators = create_model_iterators(TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE, BATCH_SIZE)

	dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = LR)

	train_model(model, iterators, dataset_sizes, BATCH_SIZE, criterion, optimizer, num_epochs = EPOCHS, device = device)

if __name__ == "__main__":
	test_model()
import torch
import torch.nn as nn

class CnnTitleClassifier(nn.Module):
	"""
	Inspired from:
	Kim, Y., Convolutional Neural Networks for Sentence Classification. Proceedings of the 2014
	Conference on Empirical Methods in Natural Language Processing (EMNLP 2014), 2014.

	https://github.com/prakashpandey9/Text-Classification-Pytorch/blob/master/models/CNN.py	
	"""
	def __init__(self, vocab_size, embedding_length):
		self.word_embeddings = nn.Embedding(vocab_size, embedding_length)
		self.word_embeddings.weight = nn.Parameter(embedding_weights, requires_grad=False)

		out_channels = 50

		self.conv1 = nn.Conv2d(1, out_channels, (10, embedding_length)),
		self.conv2 = nn.Conv2d(1, out_channels, (10, embedding_length)),
		self.conv3 = nn.Conv2d(1, out_channels, (10, embedding_length)),
		self.dropout = nn.Dropout(0.5),
		self.relu = nn.ReLU()
		self.fc = nn.Linear(3 * out_channels, 30)
		self.softmax = nn.Softmax(0)

	def conv_block(self, input, conv):


	def forward(titles):
		input = input.unsqueeze(1)

		return self.conv_net(input)
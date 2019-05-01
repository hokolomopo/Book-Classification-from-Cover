from train_text_iterators import train_model
from cnn_text_model import create_model_iterators
import torch
import torch.nn as nn
import os
import torch.optim as optim
from sentence_encoder import *
import matplotlib.pyplot as plt

def test_batch_sizes(batch_sizes, criterion, scheduler = None, num_epochs=25, device="cpu", model_name = "cnn_text_model"):
	"""
	Test the batch size for concolutionnal network for text
	"""
	TRAIN_CSV_FILE = "dataset/train_set_cleaned.csv"
	VAL_CSV_FILE = "dataset/validation_set_cleaned.csv"
	TEST_CSV_FILE = "dataset/book30-listing-test_cleaned.csv"

	try:
		os.mkdir("plots_text_model")
	except:
		pass

	LR = 0.001

	PLOT_DIR = "plots_text_model/"
	MODEL_DIR = "text_models/"

	file_name = model_name + "_batch"

	for batch_size in batch_sizes:
		model, iterators = create_model_iterators(TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE, batch_size)
		dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}
		optimizer = optim.Adam(model.parameters(), lr = LR)
		model.to(device)
		print("batch_size = {}".format(batch_size))
		model, stats, lrstats = train_model(model, data_loaders, dataset_sizes, batch_size, criterion, 
											optimizer, scheduler, num_epochs, device, scheduler_step = "cycle")
		torch.save(model.state_dict(),MODEL_DIR + model_name + "_batch_" + str(batch_size) + ".pt")
		plt.plot(stats.epochs['val'],  stats.accuracies['val'], label="batch size {}".format(batch_size))
		file_name += "_{}".format(batch_size)

	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.legend()
	plt.savefig(PLOT_DIR + file_name + ".pdf")

if __name__ == "__main__":
	EPOCHS = 2
	
	BATCH_SIZES = (4, 8, 16, 32, 64)

	criterion = nn.CrossEntropyLoss()
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	test_batch_sizes(BATCH_SIZES, criterion, num_epochs = EPOCHS, device = device)
	
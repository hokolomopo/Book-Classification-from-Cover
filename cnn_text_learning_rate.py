from train_text_iterators import train_model
from cnn_text_model import create_model_iterators
import torch.nn as nn
import torch.optim as optim
from sentence_encoder import *
import cyclic_sceduler
import matplotlib.pyplot as plt
import os

def test_learning_rates(model, iterators, batch_size, criterion, num_epochs = 25, device = "cpu", prefix = "cnn_"):
	"""
	Test learning rate for title classification with convolutionnal networks
	for cyclic learning rate
	"""
	MIN_LR = 0.0001
	MAX_LR = 0.1

	try:
		os.mkdir("plots_text_model")
	except:
		pass

	PLOT_DIR = "plots_text_model/"

	dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}

	optimizer = optim.SGD(model.parameters(), lr = MIN_LR, momentum = 0.9)
	exp_lr_scheduler = cyclic_sceduler.CyclicLR(optimizer, base_lr = MIN_LR, max_lr = MAX_LR, 
														 step_size = num_epochs * dataset_sizes['train'] / batch_size)

	model, stats , lrstats = train_model(model, iterators, dataset_sizes, batch_size, criterion, optimizer, exp_lr_scheduler , 
										 num_epochs = num_epochs, device = device, scheduler_step = "batch")

	lrs, accs = zip(*lrstats)

	plt.figure(frameon  = False)
	plt.plot(lrs,  accs)
	plt.xlabel('Learning Rate')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + prefix + "text_learning_rate.pdf")

def test_learning_rates_adam(model, iterators, batch_size, criterion, learning_rates, num_epochs = 25, device = "cpu", prefix = "cnn_"):
	"""
	Test learning rate for title classification with convolutionnal networks
	for adam
	"""
	try:
		os.mkdir("plots_text_model")
	except:
		pass

	PLOT_DIR = "plots_text_model/"

	dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}

	for learning_rate in learning_rates:
		optimizer = optim.Adam(model.parameters(), lr = learning_rate)
		model, stats , lrstats = train_model(model, iterators, dataset_sizes, batch_size, criterion, optimizer, 
										     num_epochs = num_epochs, device = device, scheduler_step = "batch")
		plt.plot(stats.epochs['val'],  stats.accuracies['val'], label = "{}".format(learning_rate))

	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.legend()
	plt.savefig(PLOT_DIR + prefix + "learning_rate_adam.pdf")

if __name__ == "__main__":
	TRAIN_CSV_FILE = "dataset/train_set_cleaned.csv"
	VAL_CSV_FILE = "dataset/validation_set_cleaned.csv"
	TEST_CSV_FILE = "dataset/book30-listing-test_cleaned.csv"

	NB_INPUTS = 4096
	NB_OUTPUTS = 30
	EPOCHS = 100
	BATCH_SIZE = 64

	model, iterators = create_model_iterators(TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE, BATCH_SIZE)
	criterion = nn.CrossEntropyLoss()

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	test_learning_rates(model, iterators, BATCH_SIZE, criterion, num_epochs = EPOCHS, device = device)

	LEARNING_RATES = [0.00001, 0.0001, 0.001, 0.01, 0.1]
	test_learning_rates_adam(model, iterators, BATCH_SIZE, criterion, LEARNING_RATES, num_epochs = EPOCHS, device = device)
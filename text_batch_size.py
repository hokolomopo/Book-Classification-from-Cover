from learning_rate_cyclic import train_model
import torch
import torch.nn as nn
import os
import torch.optim as optim
from sentence_encoder import *
import cyclic_sceduler
import matplotlib.pyplot as plt
from text_model import create_model_3, load_data_loaders

def test_batch_sizes(model, batch_sizes, criterion, optimizer, scheduler = None, num_epochs=25, device="cpu", model_name = "text_model", glove = False):
	"""
	Test different batch sizes for title classification using InferSent and Adam optimizer
	"""
	try:
		os.mkdir("plots_text_model")
	except:
		pass

	PLOT_DIR = "plots_text_model/"
	MODEL_DIR = "text_models/"

	file_name = model_name + "_batch"

	for batch_size in batch_sizes:
		if glove:
			data_loaders_file = "dataloaders/encoded_text_data_loaders_glove_{}.pickle".format(batch_size)
		else:
			data_loaders_file = "dataloaders/encoded_text_data_loaders_{}.pickle".format(batch_size)
		data_loaders = load_data_loaders(data_loaders_file)
		dataset_sizes = {phase: len(data_loader.dataset) for phase, data_loader in data_loaders.items()}
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

def test_batch_sizes_cyclic_lr(model, batch_sizes, criterion, optimizer, min_lr, max_lr, cycle_length, num_epochs=25, device="cpu", model_name = "text_model", glove = False):
	"""
	Test different batch sizes for title classification using InferSent and cyclic learning rate
	"""
	try:
		os.mkdir("plots_text_model")
	except:
		pass

	PLOT_DIR = "plots_text_model/"
	MODEL_DIR = "text_models/"

	file_name = model_name + "_batch_cyclic"

	for batch_size in batch_sizes:
		if glove:
			data_loaders_file = "dataloaders/encoded_text_data_loaders_glove_{}.pickle".format(batch_size)
		else:
			data_loaders_file = "dataloaders/encoded_text_data_loaders_{}.pickle".format(batch_size)
		data_loaders = load_data_loaders(data_loaders_file)
		dataset_sizes = {phase: len(data_loader.dataset) for phase, data_loader in data_loaders.items()}
		scheduler = cyclic_sceduler.CyclicLR(optimizer, base_lr = min_lr, max_lr = max_lr, 
												  	  step_size = cycle_length * dataset_sizes['train'] / batch_size)
		print("batch_size = {}".format(batch_size))
		model, stats, lrstats = train_model(model, data_loaders, dataset_sizes, batch_size, criterion, 
											optimizer, scheduler, num_epochs, device, scheduler_step = "batch")
		torch.save(model.state_dict(),MODEL_DIR + model_name + "_batch_" + str(batch_size) + ".pt")
		plt.plot(stats.epochs['val'],  stats.accuracies['val'], label="batch size {}".format(batch_size))
		file_name += "_{}".format(batch_size)

	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.legend()
	plt.savefig(PLOT_DIR + file_name + ".pdf")

if __name__ == "__main__":
	NB_INPUTS = 4096
	NB_OUTPUTS = 30
	EPOCHS = 50
	LR = 0.001
	MIN_LR = 0.0001
	MAX_LR = 0.05
	CYCLE_LENGTH = 4

	BATCH_SIZES = (4, 8, 16, 32, 64)

	model = create_model_3(NB_INPUTS, NB_OUTPUTS)

	criterion = nn.CrossEntropyLoss()
	
	
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model.to(device)

	optimizer = optim.Adam(model.parameters(), lr = LR)
	test_batch_sizes(model, BATCH_SIZES, criterion, optimizer, num_epochs = EPOCHS, device = device)

	#optimizer = optim.SGD(model.parameters(), lr = MIN_LR, momentum = 0.9)
	#test_batch_sizes_cyclic_lr(model, BATCH_SIZES, criterion, optimizer, MIN_LR, MAX_LR, CYCLE_LENGTH, num_epochs=EPOCHS, device = device)
	
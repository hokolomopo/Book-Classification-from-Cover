from learning_rate_cyclic import train_model
from text_model import create_model_3 as create_model
from text_model import create_model_10, create_model_18, create_model_19
from text_model import load_data_loaders
import torch.nn as nn
import torch.optim as optim
from sentence_encoder import *
import cyclic_sceduler
import matplotlib.pyplot as plt
import os

def build_final_text_model_cyclic_lr():
	MIN_LR = 0.0001
	MAX_LR = 0.05
	EPOCHS = 500
	BATCH_SIZE = 64
	NB_INPUTS = 4096
	NB_OUTPUTS = 30
	CYCLE_LENGTH = 4

	try:
		os.mkdir("plots_text_model")
	except:
		pass

	try:
		os.mkdir("text_models")
	except:
	 	pass

	PLOT_DIR = "plots_text_model/"
	MODEL_DIR = "text_models/"

	dataloaders = load_data_loaders("dataloaders/encoded_text_data_loaders_{}.pickle".format(BATCH_SIZE))
	dataset_sizes = {phase: len(dataloader.dataset) for phase, dataloader in dataloaders.items()}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = create_model(NB_INPUTS, NB_OUTPUTS)

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.SGD(model.parameters(), lr = MIN_LR, momentum = 0.9)

	scheduler = cyclic_sceduler.CyclicLR(optimizer, base_lr = MIN_LR, max_lr = MAX_LR, 
												  step_size = CYCLE_LENGTH * dataset_sizes['train'] / BATCH_SIZE)

	model, stats, lrstats = train_model(model, dataloaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, scheduler, 
									     num_epochs = EPOCHS, device = device, scheduler_step = "batch", clip_gradient = True)

	plt.plot(stats.epochs['val'],  stats.accuracies['val'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "final_text_model_cyclic_lr.pdf")

	torch.save(model.state_dict(), MODEL_DIR + "final_text_model_cyclic_lr.pt")

	return model, stats, lrstats

def build_final_text_model_adam():
	LR = 0.001
	EPOCHS = 500
	BATCH_SIZE = 64
	NB_INPUTS = 4096
	NB_OUTPUTS = 30

	try:
		os.mkdir("plots_text_model")
	except:
		pass

	try:
		os.mkdir("text_models")
	except:
		pass

	PLOT_DIR = "plots_text_model/"
	MODEL_DIR = "text_models/"

	dataloaders = load_data_loaders("dataloaders/encoded_text_data_loaders_{}.pickle".format(BATCH_SIZE))
	dataset_sizes = {phase: len(dataloader.dataset) for phase, dataloader in dataloaders.items()}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = create_model(NB_INPUTS, NB_OUTPUTS)

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam(model.parameters(), lr = LR)

	model, stats, lrstats = train_model(model, dataloaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, 
									     num_epochs = EPOCHS, device = device, scheduler_step = "batch", clip_gradient = True)

	plt.plot(stats.epochs['val'],  stats.accuracies['val'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "final_text_model_adam.pdf")

	torch.save(model.state_dict(), MODEL_DIR + "final_text_model_adam.pt")

	return model, stats, lrstats

def build_10_classes_model():
	LR = 0.001
	EPOCHS = 500
	BATCH_SIZE = 4
	NB_INPUTS = 4096
	NB_OUTPUTS = 10

	try:
		os.mkdir("plots_text_model")
	except:
		pass

	try:
		os.mkdir("text_models")
	except:
		pass

	PLOT_DIR = "plots_text_model/"
	MODEL_DIR = "text_models/"

	dataloaders = load_data_loaders("dataloaders/encoded_text_data_loaders_{}_10.pickle".format(BATCH_SIZE))
	dataset_sizes = {phase: len(dataloader.dataset) for phase, dataloader in dataloaders.items()}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = create_model_19(NB_INPUTS, NB_OUTPUTS)

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam(model.parameters(), lr = LR)

	model, stats, lrstats = train_model(model, dataloaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, 
									     num_epochs = EPOCHS, device = device, scheduler_step = "batch", clip_gradient = True)

	plt.plot(stats.epochs['val'], stats.accuracies['val'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "final_text_model_10.pdf")

	torch.save(model.state_dict(), MODEL_DIR + "final_text_model_10.pt")

	return model, stats, lrstats

def compare_adam_cyclic():
	PLOT_DIR = "plots_text_model/"

	cyclic_model, cyclic_stats, cyclic_lrstats = build_final_text_model_cyclic_lr()
	adam_model, adam_stats, adam_lrstats = build_final_text_model_adam()

	plt.plot(cyclic_stats.epochs['val'], cyclic_stats.accuracies['val'], label = "cyclic_lr")
	plt.plot(adam_stats.epochs['val'], adam_stats.accuracies['val'], label = "adam")
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.legend()
	plt.savefig(PLOT_DIR + "compare_text_cyclic_adam.pdf")

if __name__ == "__main__":
	#build_final_text_model_cyclic_lr()
	#build_final_text_model_adam()
	#compare_adam_cyclic()
	build_10_classes_model()
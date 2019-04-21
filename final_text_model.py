from learning_rate_cyclic import train_model
from text_model import create_model_3 as create_model
from text_model import load_data_loaders
import torch.nn as nn
import torch.optim as optim
from sentence_encoder import *
import liboptim.cyclic_sceduler
import matplotlib.pyplot as plt

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

	scheduler = liboptim.cyclic_sceduler.CyclicLR(optimizer, base_lr = MIN_LR, max_lr = MAX_LR, 
												  step_size = CYCLE_LENGTH * dataset_sizes['train'] / BATCH_SIZE)

	model, stats, lrstats = train_model(model, dataloaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, scheduler, 
									     num_epochs = EPOCHS, device = device, scheduler_step = "batch")

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

	dataloaders = load_data_loaders("dataloaders/final_encoded_text_data_loaders_{}.pickle".format(BATCH_SIZE))
	dataset_sizes = {phase: len(dataloader.dataset) for phase, dataloader in dataloaders.items()}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model = create_model(NB_INPUTS, NB_OUTPUTS)

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam(model.parameters(), lr = LR)

	model, stats, lrstats = train_model(model, dataloaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, 
									     num_epochs = EPOCHS, device = device, scheduler_step = "batch")

	plt.plot(stats.epochs['val'],  stats.accuracies['val'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "final_text_model_adam.pdf")

	torch.save(model.state_dict(), MODEL_DIR + "final_text_model_adam.pt")

	return model, stats, lrstats


if __name__ == "__main__":
	#build_final_text_model_cyclic_lr()
	build_final_text_model_adam()
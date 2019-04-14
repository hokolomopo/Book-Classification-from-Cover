from learning_rate_cyclic import train_model
from text_model import create_model_2 as create_model
from text_model import load_data_loaders
import torch.nn as nn
import torch.optim as optim
from sentence_encoder import *
import liboptim.cyclic_sceduler
import matplotlib.pyplot as plt

def test_learning_rates(model, dataloaders, batch_size, criterion, num_epochs = 25, device = "cpu"):
	MIN_LR = 0.0001
	MAX_LR = 0.002

	PLOT_DIR = "plots_text_model/"

	dataset_sizes = {phase: len(dataloader.dataset) for phase, dataloader in dataloaders.items()}

	optimizer = optim.SGD(model.parameters(), lr = MIN_LR, momentum = 0.9)
	exp_lr_scheduler = liboptim.cyclic_sceduler.CyclicLR(optimizer, base_lr = MIN_LR, max_lr = MAX_LR, 
														 step_size = num_epochs * dataset_sizes['train'] / batch_size)

	model, stats , lrstats = train_model(model, dataloaders, dataset_sizes, batch_size, criterion, optimizer, exp_lr_scheduler , 
										 num_epochs = num_epochs, device = device, scheduler_step = "batch")

	lrs, accs = zip(*lrstats)

	plt.figure(frameon  = False)
	plt.plot(lrs,  accs)
	plt.xlabel('Learning Rate')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "text_learning_rate.pdf")

if __name__ == "__main__":
	NB_INPUTS = 4096
	NB_OUTPUTS = 30
	EPOCHS = 2
	BATCH_SIZE = 4

	model = create_model(NB_INPUTS, NB_OUTPUTS)
	criterion = nn.CrossEntropyLoss()

	dataloaders = load_data_loaders("dataloaders/encoded_text_data_loaders_{}.pickle".format(BATCH_SIZE))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	test_learning_rates(model, dataloaders, BATCH_SIZE, criterion, num_epochs = EPOCHS, device = device)

from text_model import *
from testmodel import *
import torch
import torch.nn as nn
import os

def test_batch_sizes(model, batch_sizes, criterion, optimizer, scheduler = None, num_epochs=25, device="cpu", model_name = "text_model"):
	try:
		os.mkdir("plots_text_model")
	except:
		pass

	PLOT_DIR = "plots_text_model/"
	MODEL_DIR = "text_models/"

	file_name = model_name + "_batch"
	for batch_size in batch_sizes:
		data_loaders_file = "dataloaders/encoded_text_data_loaders_{}.pickle".format(batch_size)
		data_loaders = load_data_loaders(data_loaders_file)
		dataset_sizes = {phase: len(data_loader.dataset) for phase, data_loader in data_loaders.items()}
		print("batch_size = {}".format(batch_size))
		model, stats = train_model(model, data_loaders, dataset_sizes, batch_size, criterion, optimizer, scheduler, num_epochs, device)
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
	EPOCHS = 20
	LR = 0.0001

	BATCH_SIZES = (4, 8, 16, 32, 64)

	model = create_model_2(NB_INPUTS, NB_OUTPUTS)

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.0001)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	test_batch_sizes(model, BATCH_SIZES, criterion, optimizer, num_epochs = EPOCHS, device = device)
	
from sentence_encoder import *
from testmodel import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler

def create_model(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(4096, 300),
		nn.Linear(300, 300),
		nn.Linear(300, nb_outputs),
		nn.Softmax(0)
		)

	return model

def test_text_model(model, data_loaders_file):
	BATCH_SIZE = 4
	EPOCHS = 20

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	print("load dataloaders")
	data_loaders = pickle.load(open(data_loaders_file, "rb"))
	dataset_sizes = {phase: len(data_loader.dataset) for phase, data_loader in data_loaders.items()}

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.0001)

	print("train")
	model, stats = train_model(model, data_loaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, num_epochs = EPOCHS, device = device)

	print(stats)

	torch.save(model.state_dict(), "text_model.pt")

if __name__ == "__main__":
	model = create_model(4096, 30)
	data_loaders_file = "text_data_loaders.pickle"

	test_text_model(model, data_loaders_file)
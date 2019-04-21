from testmodel import load_resnet, change_model
import torch.nn as nn
import torch
from text_model import create_model_3, load_data_loaders
from learning_rate_cyclic import train_model
from combined_dataloaders import CombinedBookDataset
import torch.optim as optim
import matplotlib.pyplot as plt

class CombinedModel(nn.Module):
	def __init__(self):
		super().__init__()
		resnet = 18
		trained_layers = 10 
		n_outputs = 30

		self.image_model = load_resnet(resnet)
		self.image_model = change_model(self.image_model, trained_layers, n_outputs)
		self.image_model.load_state_dict(torch.load("cover_final/64 w relu/model64"))
		removed = list(self.image_model.fc.children())[:-2]
		self.image_model.fc = nn.Sequential(*removed)

		self.text_model = create_model_3(4096, n_outputs)
		self.text_model.load_state_dict(torch.load("text_models/final_text_model.pt"))
		removed = list(self.text_model.children())[:-2]
		self.text_model = nn.Sequential(*removed)

		self.join_layer = nn.Sequential(nn.Linear(1256, 1256),
										nn.LeakyReLU(),
										nn.Linear(1256, n_outputs),
										nn.Softmax(0),
										)

	def forward(self, inputs):
		cover = inputs[0]
		title_emb = inputs[1]

		cover_output = self.image_model(cover)
		title_output = self.text_model(title_emb)

		merged_output = torch.cat((cover_output, title_output), 1)

		return self.join_layer(merged_output)

def test_combined_model():
	BATCH_SIZE = 32
	EPOCHS = 5
	LR = 0.001
	MODEL_DIR = "combined_models/"
	PLOT_DIR = "plots_combined_model/"

	model = CombinedModel()
	data_loaders = load_data_loaders("dataloaders/combined_data_loaders_{}.pickle".format(BATCH_SIZE))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	dataset_sizes = {phase: len(data_loader.dataset) for phase, data_loader in data_loaders.items()}

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = LR)

	print("train")
	model, stats, lrstats = train_model(model, data_loaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, num_epochs = EPOCHS, device = device)

	print(stats)

	torch.save(model.state_dict(), MODEL_DIR + "combined_model.pt")

	plt.plot(stats.epochs['val'],  stats.accuracies['val'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "combined_model.pdf")

	return model, stats, lrstats

if __name__ == "__main__":
	test_combined_model()
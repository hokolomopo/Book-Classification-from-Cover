from testmodel import load_resnet, change_model
import torch.nn as nn
import torch
from train_text_iterators import train_model
from cnn_combined_data import create_combined_text_iterators
import torch.optim as optim
import matplotlib.pyplot as plt
from cnn_text_model import CnnTitleClassifier

class IdentityModule(nnModule):
	"""
	Module that performs the identity function
	"""
    def forward(self, inputs):
        return inputs

class CombinedModel(nn.Module):
	"""
	model combining covers and text with the convolutionnal network
	"""
	def __init__(self, text_model):
		super().__init__()
		resnet = 18
		trained_layers = 10 
		n_outputs = 30

		self.image_model = load_resnet(resnet)
		self.image_model = change_model(self.image_model, trained_layers, n_outputs)
		self.image_model.load_state_dict(torch.load("cover_final/64 w relu/model64"))
		removed = list(self.image_model.fc.children())[:-2]
		self.image_model.fc = nn.Sequential(*removed)

		self.text_model = text_model
		self.text_model.fc = IdentityModule()
		
		self.join_layer = nn.Sequential(nn.Linear(3 * self.text_model.out_channels + 256, n_outputs),
										nn.Softmax(0)
										)

	def forward(self, inputs):
		cover = inputs[0]
		title_emb = inputs[1]

		cover_output = self.image_model(cover)
		title_output = self.text_model(title_emb)

		merged_output = torch.cat((cover_output, title_output), 1)

		return self.join_layer(merged_output)

def create_combined_model_iterators(train_csv_file, val_csv_file, test_csv_file, batch_size):
	"""
	Create iterators for combination of cover and title for convolutionnal network
	"""
	
	EMBEDDING_LENGTH = 300

	TITLE, word_embedding, iterators = create_combined_text_iterators(train_csv_file, val_csv_file, test_csv_file, batch_size, num_workers = 0)
	text_model = CnnTitleClassifier(len(TITLE.vocab), EMBEDDING_LENGTH, word_embedding)
	text_model.load_state_dict(torch.load("text_models/cnn_final_text_model_adam.pt"))

	model = CombinedModel(text_model)

	return model, iterators

def test_combined_model():
	"""
	Test the model
	"""

	BATCH_SIZE = 32
	EPOCHS = 5
	LR = 0.001
	MODEL_DIR = "combined_models/"
	PLOT_DIR = "plots_combined_model/"

	model, iterators = create_combined_model_iterators(TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE, BATCH_SIZE)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = LR)

	print("train")
	model, stats, lrstats = train_model(model, data_loaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, num_epochs = EPOCHS, device = device, combined = True)

	print(stats)

	torch.save(model.state_dict(), MODEL_DIR + "cnn_combined_model.pt")

	plt.plot(stats.epochs['val'],  stats.accuracies['val'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "cnn_combined_model.pdf")

	return model, stats, lrstats

if __name__ == "__main__":
	test_combined_model()
from sentence_encoder import *
from testmodel import *
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from sentence_encoder import SentenceEmbedding

def create_model_1(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 300),
		nn.Linear(300, 300),
		nn.Linear(300, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_2(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.Linear(1000, 1000),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_3(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.Linear(1000, 1000),
		nn.Dropout(0.5),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_4(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_5(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Linear(200, 200),
		nn.Dropout(0.5),
		nn.Linear(200, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_6(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.Linear(1000, 1000),
		nn.Linear(1000, 1000),
		nn.Linear(1000, 1000),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_7(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.Linear(1000, 1000),
		nn.Linear(1000, 1000),
		nn.Linear(1000, 1000),
		nn.Dropout(0.5),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_8(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 3000),
		nn.Linear(3000, 3000),
		nn.Linear(3000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_9(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 3000),
		nn.Linear(3000, 3000),
		nn.Dropout(0.5),
		nn.Linear(3000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_10(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, nb_outputs),
		nn.Softmax(0)
		)

	return model

def load_data_loaders(data_loaders_file):
	print("load dataloaders")
	data_loaders = pickle.load(open(data_loaders_file, "rb"))

	return data_loaders

def test_text_model(model, data_loaders, model_num = ""):
	BATCH_SIZE = 4
	EPOCHS = 10

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	
	dataset_sizes = {phase: len(data_loader.dataset) for phase, data_loader in data_loaders.items()}

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.0001)

	print("train")
	model, stats = train_model(model, data_loaders, dataset_sizes, BATCH_SIZE, criterion, optimizer, num_epochs = EPOCHS, device = device)

	print(stats)

	torch.save(model.state_dict(), "text_model"+ model_num + ".pt")

	return stats

def compare_models(nb_inputs, nb_outputs, data_loaders_file):
	try:
		os.mkdir("plots_text_model")
	except:
		pass

	PLOT_DIR = "plots_text_model/"

	data_loaders = load_data_loaders(data_loaders_file)

	models = [create_model_2(nb_inputs, nb_outputs),
			  create_model_8(nb_inputs, nb_outputs),
			  create_model_9(nb_inputs, nb_outputs),
			  create_model_10(nb_inputs, nb_outputs)
			 ]

	model_nums = ["2", "8", "9", "10"]
	title = "Compare models"
	file_name = "compare_models"

	for model, model_num in zip(models, model_nums):
		print("model {}".format(model_num))
		stats = test_text_model(model, data_loaders, model_num)
		plt.plot(stats.epochs['val'],  stats.accuracies['val'], label="model {}".format(model_num))
		title += " {}".format(model_num)
		file_name += "_{}".format(model_num)

	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.title(title)
	plt.grid(True)
	plt.legend()
	plt.savefig(PLOT_DIR + file_name + ".pdf")

if __name__ == "__main__":
	NB_INPUTS = 4096
	NB_OUTPUTS = 30

	DATA_LOADERS_FILE = "dataloaders/text_data_loaders.pickle"
	
	compare_models(NB_INPUTS, NB_OUTPUTS, DATA_LOADERS_FILE)

	
	
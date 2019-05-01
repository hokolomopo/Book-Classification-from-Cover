from sentence_encoder import *
from learning_rate_cyclic import train_model
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import os
from sentence_encoder import SentenceEmbedding
from matplotlib import pyplot as plt

"""
Define several models to try to put on top of Infersent
"""

def create_model_1(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 300),
		nn.ReLU(),
		nn.Linear(300, 300),
		nn.ReLU(),
		nn.Linear(300, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_2(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_3(nb_inputs, nb_outputs, dropout = 0.5):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Dropout(dropout),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_4(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_5(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Linear(300, 300),
		nn.LeakyReLU(),
		nn.Dropout(0.5),
		nn.Linear(300, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_6(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_7(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Linear(1000, 1000),
		nn.ReLU(),
		nn.Dropout(0.5),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_8(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 3000),
		nn.ReLU(),
		nn.Linear(3000, 3000),
		nn.ReLU(),
		nn.Linear(3000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_9(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 3000),
		nn.ReLU(),
		nn.Linear(3000, 3000),
		nn.ReLU(),
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

def create_model_11(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_12(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Dropout(0.5),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def init_weights(module):
	if type(module) == nn.Linear:
		nn.init.xavier_uniform(module.weight)

def create_model_13(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	model.apply(init_weights)
	return model

def create_model_14(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Linear(1000, 1000),
		nn.LeakyReLU(),
		nn.Dropout(0.5),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	model.apply(init_weights)
	return model

def create_model_15(nb_inputs, nb_outputs):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.ReLU(),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_16(nb_inputs, nb_outputs, dropout = 0.5):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 1000),
		nn.ReLU(),
		nn.Dropout(dropout),
		nn.Linear(1000, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_17(nb_inputs, nb_outputs, dropout = 0.5):
	model = nn.Sequential(
		nn.Dropout(dropout),
		nn.Linear(nb_inputs, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_18(nb_inputs, nb_outputs, dropout = 0.5):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 300),
		nn.ReLU(),
		nn.Linear(300, 300),
		nn.ReLU(),
		nn.Dropout(dropout),
		nn.Linear(300, nb_outputs),
		nn.Softmax(0)
		)

	return model

def create_model_19(nb_inputs, nb_outputs, dropout = 0.5):
	model = nn.Sequential(
		nn.Linear(nb_inputs, 100),
		nn.ReLU(),
		nn.Linear(100, 100),
		nn.ReLU(),
		nn.Dropout(dropout),
		nn.Linear(100, nb_outputs),
		nn.Softmax(0)
		)

	return model

def load_data_loaders(data_loaders_file):
	"""
	Load saved dataloaders
	"""
	print("load dataloaders")
	data_loaders = pickle.load(open(data_loaders_file, "rb"))

	return data_loaders

def test_text_model(model, data_loaders, batch_size, epochs, model_num = "", clip_gradient = False, print_grad = False):
	"""
	Test a model
	"""
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	MODEL_DIR = "text_models/"
	
	dataset_sizes = {phase: len(data_loader.dataset) for phase, data_loader in data_loaders.items()}

	criterion = nn.CrossEntropyLoss()
	optimizer = optim.Adam(model.parameters(), lr = 0.001)

	print("train")
	model, stats, lrstats = train_model(model, data_loaders, dataset_sizes, batch_size, criterion, optimizer, 
										num_epochs = epochs, device = device, clip_gradient = clip_gradient,
										print_grad = print_grad)

	print(stats)

	torch.save(model.state_dict(), MODEL_DIR + "text_model"+ model_num + ".pt")

	return stats

def compare_models(nb_inputs, nb_outputs):
	"""
	Compare different models
	"""
	try:
		os.mkdir("plots_text_model")
	except:
		pass

	PLOT_DIR = "plots_text_model/"
	
	BATCH_SIZE = 64
	EPOCHS = 200

	data_loaders_file = "dataloaders/encoded_text_data_loaders_{}.pickle".format(BATCH_SIZE)

	data_loaders = load_data_loaders(data_loaders_file)
	
	models = [(create_model_10(nb_inputs, nb_outputs), "10", "no hidden layer", True),
			  (create_model_17(nb_inputs, nb_outputs), "17", "no hidden layer dropout", True),
			  (create_model_15(nb_inputs, nb_outputs), "15", "one hidden layer", True),
			  (create_model_16(nb_inputs, nb_outputs), "16", "one hidden layer dropout", True),
			  (create_model_2(nb_inputs, nb_outputs), "2", "shallow", True),
			  (create_model_3(nb_inputs, nb_outputs), "3", "shallow dropout", True)
			 ]
	

	title = "Compare models"
	file_name = "compare_models_clip_gradient"

	for model, model_num, model_name, clip_gradient in models:
		print("model {}".format(model_num))
		stats = test_text_model(model, data_loaders, BATCH_SIZE, EPOCHS, model_num, clip_gradient = clip_gradient)
		plt.plot(stats.epochs['val'],  stats.accuracies['val'], label= model_name)
		file_name += "_{}".format(model_num)

	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.legend()
	plt.savefig(PLOT_DIR + file_name + ".pdf")

def compare_dropout(nb_inputs, nb_outputs):
	"""
	Compare drop out values
	"""
	try:
		os.mkdir("plots_text_model")
	except:
		pass

	PLOT_DIR = "plots_text_model/"

	BATCH_SIZE = 64
	EPOCHS = 100

	data_loaders_file = "dataloaders/encoded_text_data_loaders_{}.pickle".format(BATCH_SIZE)

	data_loaders = load_data_loaders(data_loaders_file)

	dropouts = [x/10 for x in range(1,10)]

	for dropout in dropouts:
		print("dropout = {}".format(dropout))
		model = create_model_3(nb_inputs, nb_outputs, dropout)
		stats = test_text_model(model, data_loaders, BATCH_SIZE, EPOCHS)
		plt.plot(stats.epochs['val'],  stats.accuracies['val'], label = "dropout = {}".format(dropout))

	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.legend()
	plt.savefig(PLOT_DIR + "compare_dropout.pdf")

def perform_test():
	"""
	Test a model
	"""
	NB_INPUTS = 4096
	NB_OUTPUTS = 30
	BATCH_SIZE = 64
	EPOCHS = 5

	data_loaders_file = "dataloaders/encoded_text_data_loaders_{}.pickle".format(BATCH_SIZE)

	data_loaders = load_data_loaders(data_loaders_file)
	model = create_model_12(NB_INPUTS, NB_OUTPUTS)

	test_text_model(model, data_loaders, BATCH_SIZE, EPOCHS, clip_gradient = True, print_grad = True)

if __name__ == "__main__":
	NB_INPUTS = 4096
	NB_OUTPUTS = 30
	
	perform_test()
	#compare_models(NB_INPUTS, NB_OUTPUTS)
	#compare_dropout(NB_INPUTS, NB_OUTPUTS)

	
	
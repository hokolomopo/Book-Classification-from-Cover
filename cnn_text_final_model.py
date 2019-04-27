from train_text_iterators import train_model
from cnn_text_model import create_model_iterators
import torch.nn as nn
import torch.optim as optim
from sentence_encoder import *
import cyclic_sceduler
import matplotlib.pyplot as plt

def build_final_text_model_cyclic_lr():
	TRAIN_CSV_FILE = "dataset/train_set_cleaned.csv"
	VAL_CSV_FILE = "dataset/validation_set_cleaned.csv"
	TEST_CSV_FILE = "dataset/book30-listing-test_cleaned.csv"

	MIN_LR = 0.0001
	MAX_LR = 0.03
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

	model, iterators = create_model_iterators(TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE, BATCH_SIZE)
	dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.SGD(model.parameters(), lr = MIN_LR, momentum = 0.9)

	scheduler = cyclic_sceduler.CyclicLR(optimizer, base_lr = MIN_LR, max_lr = MAX_LR, 
												  step_size = CYCLE_LENGTH * dataset_sizes['train'] / BATCH_SIZE)

	model, stats, lrstats = train_model(model, iterators, dataset_sizes, BATCH_SIZE, criterion, optimizer, scheduler, 
									     num_epochs = EPOCHS, device = device, scheduler_step = "batch")

	plt.plot(stats.epochs['val'],  stats.accuracies['val'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "cnn_final_text_model_cyclic_lr.pdf")
	plt.close()

	torch.save(model.state_dict(), MODEL_DIR + "cnn_final_text_model_cyclic_lr.pt")

	return model, stats, lrstats

def build_final_text_model_adam():
	TRAIN_CSV_FILE = "dataset/train_set_cleaned.csv"
	VAL_CSV_FILE = "dataset/validation_set_cleaned.csv"
	TEST_CSV_FILE = "dataset/book30-listing-test_cleaned.csv"

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

	model, iterators = create_model_iterators(TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE, BATCH_SIZE)
	dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	criterion = nn.CrossEntropyLoss()

	optimizer = optim.Adam(model.parameters(), lr = LR)

	model, stats, lrstats = train_model(model, iterators, dataset_sizes, BATCH_SIZE, criterion, optimizer, 
									     num_epochs = EPOCHS, device = device, scheduler_step = "batch")

	plt.plot(stats.epochs['val'],  stats.accuracies['val'])
	plt.xlabel('Epochs')
	plt.ylabel('Accuracy')
	plt.grid(True)
	plt.savefig(PLOT_DIR + "cnn_final_text_model_adam.pdf")
	plt.close()

	torch.save(model.state_dict(), MODEL_DIR + "cnn_final_text_model_adam.pt")

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
	plt.savefig(PLOT_DIR + "cnn_compare_text_cyclic_adam.pdf")
	plt.close()

if __name__ == "__main__":
	#build_final_text_model_cyclic_lr()
	#build_final_text_model_adam()
	compare_adam_cyclic()
from cover_test_model import print_acc
from text_model import load_data_loaders
from combined_model import CombinedModel
from combined_dataloaders import *
import sys
import torch

def test_combined_model(topK):
	"""
	Test the combined model on the test set
	"""
	BATCH_SIZE = 32
	print("creating model")
	model = CombinedModel()
	model.load_state_dict(torch.load("combined_models/combined_model.pt"))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	print("loading dataloaders")
	dataloaders = load_data_loaders("dataloaders/combined_data_loaders_{}.pickle".format(BATCH_SIZE))
	test_dataloader = dataloaders["test"]

	dataset_size = len(test_dataloader.dataset)

	print("computing acc")
	print_acc(model, test_dataloader, dataset_size, topK, BATCH_SIZE, device)

def test_combined_model_10(topK):
	"""
	Test the combined model on the test set with 10 classes
	"""
	BATCH_SIZE = 32
	print("creating model")
	model = CombinedModel(10)
	model.load_state_dict(torch.load("combined_models/combined_model_10.pt"))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	print("loading dataloaders")
	dataloaders = load_data_loaders("dataloaders/combined_data_loaders_{}_10.pickle".format(BATCH_SIZE))
	test_dataloader = dataloaders["test"]

	dataset_size = len(test_dataloader.dataset)

	print("computing acc")
	print_acc(model, test_dataloader, dataset_size, topK, BATCH_SIZE, device)

if __name__ == "__main__":
	topK = int(sys.argv[1])
	test_combined_model_10(topK)
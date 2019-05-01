from cover_test_model import print_acc
from text_model import create_model_3, create_model_19, load_data_loaders
import sys
import torch
from sentence_encoder import *

def test_text_model(topK):
	BATCH_SIZE = 64
	print("creating model")
	model = create_model_3(4096, 30)
	model.load_state_dict(torch.load("text_models/final_text_model_adam.pt"))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	print("loading dataloaders")
	dataloaders = load_data_loaders("dataloaders/encoded_text_data_loaders_{}.pickle".format(BATCH_SIZE))
	test_dataloader = dataloaders["test"]

	dataset_size = len(test_dataloader.dataset)

	print("computing acc")
	print_acc(model, test_dataloader, dataset_size, topK, BATCH_SIZE, device)

def test_text_model_10(topK):
	BATCH_SIZE = 64
	print("creating model")
	model = create_model_19(4096, 10)
	model.load_state_dict(torch.load("text_models/final_text_model_10.pt"))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	print("loading dataloaders")
	dataloaders = load_data_loaders("dataloaders/encoded_text_data_loaders_{}_10.pickle".format(BATCH_SIZE))
	test_dataloader = dataloaders["test"]

	dataset_size = len(test_dataloader.dataset)

	print("computing acc")
	print_acc(model, test_dataloader, dataset_size, topK, BATCH_SIZE, device)

if __name__ == "__main__":
	topK = int(sys.argv[1])
	test_text_model_10(topK)
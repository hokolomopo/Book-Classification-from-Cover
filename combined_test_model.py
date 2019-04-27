from cover_test_model import print_acc
from text_model import load_data_loaders
from combined_model import CombinedModel
import sys
import torch

def test_text_model(topK):
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

if __name__ == "__main__":
	topK = int(sys.argv[1])
	test_text_model(topK)
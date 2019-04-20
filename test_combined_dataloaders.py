from combined_dataloaders import *
from testmodel import load_resnet, change_model

if __name__ == "__main__":

	data_loaders = pickle.load(open("dataloaders/combined_data_loaders_4.pickle", "rb"))
	print("unpickled")
	
	for input, label in data_loaders["train"]:
		print("cover = {}".format(input[0]))
		print("text = {}".format(input[1]))
		print("label = {}".format(label))

	for input, label in data_loaders["test"]:
		print("cover = {}".format(input[0]))
		print("text = {}".format(input[1]))
		print("label = {}".format(label))
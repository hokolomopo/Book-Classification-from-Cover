from sentence_encoder import *

if __name__ == "__main__":

	data_loaders = pickle.load(open("text_data_loaders.pickle", "rb"))
	print("unpickled")
	
	for input, label in enumerate(data_loaders["train"]):
		print("input = {}".format(input))
		print("label = {}".format(label))

	for input, label in enumerate(data_loaders["test"]):
		print(input)
		print(label)

from sentence_encoder import *

if __name__ == "__main__":
	train_csv_path = "dataset/book30-listing-train.csv"
	test_csv_path = "dataset/book30-listing-test.csv"
	data_loaders = create_text_data_loaders(train_csv_path, test_csv_path, 4, 4)

	for i, batch in enumerate(data_loaders["train"]):
		input = batch["title"]
		label = batch["class"]
		print("input = {}".format(input))
		print("label = {}".format(label))

	"""
	for i, batch in enumerate(data_loaders["test"]):
		input = batch["title"]
		label = batch["class"]
		print(input)
		print(label)
	"""
from cnn_combined_data import *

if __name__ == "__main__":
	train_csv_file = "dataset/train_set_cleaned.csv"
	val_csv_file = "dataset/validation_set_cleaned.csv"
	test_csv_file = "dataset/book30-listing-test_cleaned.csv"
	TITLE, word_embedding, iterators = create_combined_text_iterators(train_csv_file, val_csv_file, test_csv_file, batch_size = 4, num_workers = 0)

	for batch in iterators["train"]:
		print("title = {}".format(batch.title))
		print("cover = {}".format(batch.cover))
		print("label = {}".format(batch.label))

	for batch in iterators["val"]:
		print("title = {}".format(batch.title))
		print("cover = {}".format(batch.cover))
		print("label = {}".format(batch.label))

	for batch in iterators["test"]:
		print("title = {}".format(batch.title))
		print("cover = {}".format(batch.cover))
		print("label = {}".format(batch.label))
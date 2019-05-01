"""
Create the datasets with 10 classes from the 30 clases ones
"""

import pandas as pd

classes = ["Children's Books",
		   "Comics & Graphic Novels",
		   "Computers & Technology",
		   "Cookbooks, Food & Wine",
		   "Romance",
		   "Science & Math",
		   "Science Fiction & Fantasy",
		   "Sports & Outdoors",
		   "Test Preparation",
		   "Travel"]

datasetFiles = ["dataset/book30-listing-test", "dataset/train_set", "dataset/validation_set"]
cols = ["index", "filename", "url", "title", "author", "class", "class_name"]

for datasetFile in datasetFiles:
	print(datasetFile)

	dataset = pd.read_csv(datasetFile + ".csv", names = cols, encoding = "ISO-8859-1")
	dataset = dataset[dataset["class_name"].isin(classes)]
	dataset.to_csv(datasetFile + "_10.csv", header = False, encoding = "ISO-8859-1", index = False)

	for className in classes:
		classDataset = dataset[dataset["class_name"] == className]
		print("{}: {}".format(className, classDataset.shape[0]))


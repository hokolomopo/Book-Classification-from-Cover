from bookDataset import create_data_loaders
import torch
import torch.nn as nn
import torchvision.models as models
from matplotlib import pyplot as plt
import time

def create_model():
	"""
	resnet152 = models.resnet152(pretrained=True)
	modules = list(resnet152.children())[:-1]
	
	for module in modules:
		module.requires_grad = False
	
	modules.append(torch.nn.ReLU())
	modules.append(torch.nn.Linear(8192, 32))
	
	model = nn.Sequential(*modules)
	"""

	model = models.resnet152(pretrained = True)
	numFeatures = model.fc.in_features
	model.fc = nn.Linear(numFeatures, 32)

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)

	for param in model.parameters():
		param.requires_grad = False

	for param in model.fc.parameters():
		param.requires_grad = True

	return model

def validate_model(model, data_loaders):
	NB_EPOCHS = 50

	losses = []

	optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
	criterion = nn.CrossEntropyLoss()
	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

	model.train()

	for epoch in range(NB_EPOCHS):
		start = time.time()

		print("epoch " , epoch)
		for i, batch in enumerate(data_loaders["train"]):
			print(i)
			input = batch["cover"]
			label = batch["class"]

			input = input.to(device)
			label = label.to(device)

			pred = model(input)

			loss = criterion(pred, label)
			losses.append(loss)

			optimizer.zero_grad()
			loss.backward()
			optimizer.step()
		end = time.time()
		print("Time for epoch: " ,end - start)


	# Show the loss over the training iterations.
	plt.plot(losses, color="black")
	plt.minorticks_on()
	plt.xlabel("Iterations")
	plt.ylabel("Loss")
	plt.grid(True, alpha=.2)
	plt.savefig("test.png")

if __name__ == "__main__":
	train_csv_path = "dataset/book30-listing-train.csv"
	test_csv_path = "dataset/book30-listing-test.csv"
	cover_path = "dataset/covers"

	print("creating model...")
	model = create_model()
	print("creating loaders...")
	data_loaders = create_data_loaders(train_csv_path, test_csv_path, 
									   cover_path, 0.8, 4)
	print("validating model...")
	validate_model(model, data_loaders)


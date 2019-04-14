from text_model import *
from testmodel import *
import torch

def test_batch_sizes(model, dataloaders, dataset_sizes, batch_sizes, criterion, optimizer, scheduler = None, num_epochs=25, device="cpu", model_name = "text_model"):

	for batch_size in batch_sizes:
		model, stats = train_model(model, dataloaders, dataset_sizes, batch_size, criterion, optimizer, scheduler, num_epochs, device)
		torch.save(model.state_dict(), model_name + "_batch_" + batch_size + ".pt")
		plt.plot(stats.epochs['val'],  stats.accuracies['val'], label="batch size {}".format(batch_size))

	plt.xlabel('epoch')
	plt.ylabel('Accuracy')
	plt.title(title)
	plt.grid(True)
	plt.legend()
	plt.savefig(PLOT_DIR + file_name + ".pdf")

if __name__ == "__main__":
	EPOCHS = 10
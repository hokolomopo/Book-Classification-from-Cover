from testmodel import *
from cover_test_model import getPredInOrder, getKsAccs
from cnn_text_model import *
import sys

def print_acc(model, iterator, dataset_size, topK, batch_size, device):
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    progress = 0
    lastPrint = 0
    start = time.time()

    myAcc= 0

    for batch in iterator:
        inputs = batch.title
        labels = batch.label

        progress += batch_size / dataset_size * 100
        if(progress > 10 + lastPrint) or lastPrint == 0:
            lastPrint = progress
            print('Progress {:.2f}% time : {:.2f}'.format(progress, time.time() - start))

        if type(inputs) is list or type(inputs) is tuple:
            for i, input in enumerate(inputs):
                inputs[i] = input.to(device)
        else:
            inputs = inputs.to(device)
        
        labels = labels.to(device)
        
        # forward
        # track history if only in train
        with torch.set_grad_enabled(False):
            outputs = model(inputs)
            getPredInOrder(outputs[0])
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            myAcc += getKsAccs(outputs, labels, topK)

        # statistics
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
        

    epoch_loss = running_loss / dataset_size
    epoch_acc = running_corrects.double() / dataset_size
    epoch_acc2 = float(myAcc) / dataset_size

    print("MyAcc ", epoch_acc2)

    end = time.time()

    print('Loss: {:.4f} Acc: {:.4f}'.format(
        epoch_loss, epoch_acc))

def test_text_model(topK):
    """
    Test title classification model with convolutionnal networks on the test set
    """
	TRAIN_CSV_FILE = "dataset/train_set_cleaned.csv"
	VAL_CSV_FILE = "dataset/validation_set_cleaned.csv"
	TEST_CSV_FILE = "dataset/book30-listing-test_cleaned.csv"

	BATCH_SIZE = 32

	print("creating model")
	model, iterators = create_model_iterators(TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE, BATCH_SIZE)
	dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}

	model.load_state_dict(torch.load("text_models/cnn_final_text_model_adam.pt"))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	test_iterator = iterators["test"]
	dataset_size = dataset_sizes["test"]

	print("computing acc")
	print_acc(model, test_iterator, dataset_size, topK, BATCH_SIZE, device)

def test_text_model_10(topK):
    """
    Test title classification model with convolutionnal networks on the test set
    for the dataset with 10 classes
    """
	TRAIN_CSV_FILE = "dataset/train_set_cleaned_10.csv"
	VAL_CSV_FILE = "dataset/validation_set_cleaned_10.csv"
	TEST_CSV_FILE = "dataset/book30-listing-test_cleaned_10.csv"

	BATCH_SIZE = 32

	print("creating model")
	model, iterators = create_model_iterators(TRAIN_CSV_FILE, VAL_CSV_FILE, TEST_CSV_FILE, BATCH_SIZE)
	dataset_sizes = {key: len(iterator.data()) for key, iterator in iterators.items()}

	model.load_state_dict(torch.load("text_models/cnn_final_text_model_adam_10.pt"))

	device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
	model.to(device)
	model.eval()

	test_iterator = iterators["test"]
	dataset_size = dataset_sizes["test"]

	print("computing acc")
	print_acc(model, test_iterator, dataset_size, topK, BATCH_SIZE, device)

if __name__ == "__main__":
	topK = int(sys.argv[1])
	test_text_model_10(topK)
from cover_test_model import getPredInOrder, getKsAccs

def print_acc(model, dataloader, dataset_size, topK, batch_size, device):
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    progress = 0
    lastPrint = 0
    start = time.time()

    myAcc= 0

    for batch in iterators[phase]:
        inputs = batch.title
        labels = batch.label
        
        progress += batch_size / dataset_size * 100
        if(progress > 10 + lastPrint) or lastPrint == 0:
            lastPrint = progress
            print('Progress {:.2f}% time : {:.2f}'.format(progress, time.time() - start))

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
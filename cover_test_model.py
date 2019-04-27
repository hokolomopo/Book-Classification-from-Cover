from testmodel import *

def getPredInOrder(preds):
    res = []

    cpupred = preds.cpu()
    for i in range(len(preds)):
        _ , best = torch.kthvalue(cpupred,  len(cpupred) - i, 0)
        res.append(int(best))

    return res

def getKsAccs(preds, labels, topK):
    labels = labels.cpu()
    labels = np.asarray(labels)
    labels = labels.tolist()

    predInOrder = []
    for i in range(len(preds)):
        tmp = getPredInOrder(preds[i])
        predInOrder.append(tmp)

    accs= []
    for i in range(30):
        accs.append(0)


    for i in range(len(preds)):
        for j in range(len(preds[i])):
            if(predInOrder[i][j] == labels[i]):
                accs[j] += 1

    cumsum = 0
    for i in range(len(accs)):
        acc = accs[i] 
        accs[i] += cumsum
        cumsum += acc

    return accs[topK - 1]

def print_acc(model, dataloader, dataset_size, topK, batch_size, device):
    criterion = nn.CrossEntropyLoss()

    running_loss = 0.0
    running_corrects = 0

    # Iterate over data.
    progress = 0
    lastPrint = 0
    start = time.time()

    myAcc= 0

    for inputs, labels in dataloader:
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


if __name__ == "__main__":
    batch_size = 64
    n_workers = 2

    min_lr = 1e-4
    max_lr = 6e-3

    modelPath = "cover_final/64 w relu/model64"

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print("Loading model...")
    model = load_resnet(18)
    model = change_model(model, 10, 30)
    model = model.to(device)

    model.load_state_dict(torch.load(modelPath))
    model.eval()
    print("Loaded !")


    transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    cover_path = "dataset/covers"
    csv_path = "dataset/book30-listing-test.csv"


    image_dataset = BookDataset(csv_path, cover_path, transform=transform)
    dataloader = torch.utils.data.DataLoader(image_dataset, batch_size=batch_size,
        shuffle=True, num_workers=n_workers, pin_memory=False)

    dataset_size = len(image_dataset)
    class_names = image_dataset.classes

    print_acc(model, dataloader, dataset_size, 0, batch_size, device)
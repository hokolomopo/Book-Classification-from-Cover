from testmodel import *

"""
File used to determine teh best learning rates when using a cylci learning rate 
from the methodology proposed by the autor of the cyclic learning rate
"""


def train_model(model, dataloaders, dataset_sizes, batch_size, criterion, optimizer, scheduler = None, num_epochs=25, 
                device="cpu", scheduler_step="cycle", clip_gradient = False, print_grad = False):
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    stats = Statistics()
    lrstats = []

    model.to(device)

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if(scheduler and scheduler_step == "cycle"):
                    scheduler.step()
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            # Iterate over data.
            progress = 0
            lastPrint = 0
            start = time.time()

            for inputs, labels in dataloaders[phase]:
                progress += batch_size / dataset_sizes[phase] * 100
                if(progress > 10 + lastPrint) or lastPrint == 0:
                    lastPrint = progress
                    if scheduler:
                        print('Epoch {} : lr {}, {:.2f}% time : {:.2f}'.format(epoch, scheduler.get_lr(), progress, time.time() - start))
                    else:
                        print('Epoch {}, {:.2f}% time : {:.2f}'.format(epoch, progress, time.time() - start))

                if type(inputs) is list or type(inputs) is tuple:
                    for i, input in enumerate(inputs):
                        inputs[i] = input.to(device)
                else:
                    inputs = inputs.to(device)
                
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        if clip_gradient:
                            nn.utils.clip_grad_norm_(model.parameters(), 5)
                        if print_grad:
                            print("grad: {}".format(model[-1].weight.grad))
                        optimizer.step()
                        if(scheduler and scheduler_step == "batch"):
                            scheduler.batch_step()

                # statistics
                if type(inputs) is list or type(inputs) is tuple:
                    running_loss += loss.item() * inputs[0].size(0)
                else:
                    running_loss += loss.item() * inputs.size(0)
                    
                running_corrects += torch.sum(preds == labels.data)
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            end = time.time()

            stats.losses[phase].append(epoch_loss)
            stats.accuracies[phase].append(epoch_acc)
            stats.epochs[phase].append(epoch)
            stats.times[phase].append(end - start)

            if phase == 'val' and scheduler:
                lrstats.append((scheduler.get_lr(), epoch_acc))

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))

            print('Time taken : {}'.format(end - start))

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())


        print()


    time_elapsed = time.time() - since
    stats.time_elapsed = time_elapsed
    stats.best_acc = best_acc
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    
    # load best model weights
    model.load_state_dict(best_model_wts)

    return (model, stats, lrstats)

if __name__ == "__main__":
    # plt.ion()   # interactive mode
    n_epoch = 15
    batch_size = 64
    n_workers = 2
    resnet = 18
    trained_layers = 10 
    n_outputs = 30

    min_lr = 0.0001
    max_lr = 0.002

    # Data augmentation and normalization for training
    # Just normalization for validation
    data_transforms = {
        'train': transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

    cover_path = "dataset/covers"
    csv_paths = {'train' : "dataset/book30-listing-train.csv",
                 'val' : "dataset/book30-listing-test.csv"}


    image_datasets = {x: BookDataset(csv_paths[x], cover_path, transform=data_transforms[x])
                    for x in ['train', 'val']}
    dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=batch_size,
                                                shuffle=True, num_workers=n_workers, pin_memory=False)
                for x in ['train', 'val']}

    dataset_sizes = {x: len(image_datasets[x]) for x in ['train', 'val']}
    class_names = image_datasets['train'].classes

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    ######################################################################
    # Training the model
    # ------------------
    model_ft = load_resnet(resnet)
    model_ft = change_model(model_ft, trained_layers, n_outputs)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    stats_list = []
    folder = "cover_lr_cyclic/"

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=min_lr, momentum=0.9)

    exp_lr_scheduler = cyclic_sceduler.CyclicLR(optimizer_ft, base_lr=min_lr, max_lr=max_lr, step_size= n_epoch * dataset_sizes['train'] / batch_size)

    model_ft, stats , lrstats = train_model(model_ft, dataloaders, dataset_sizes, batch_size, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=n_epoch, device=device, scheduler_step="batch")

    stats_list.append(stats)

    lr = -1
    file = open(folder + "lr_results.txt", "a+")
    file.write("Resnet{}, lr : {}, batch size : {}, trained_layers : {}, n_outputs : {}\n".format(resnet, lr, batch_size, trained_layers, n_outputs))

    for phase in ['train', 'val']:
        for i in range(len(stats.epochs[phase])):
            file.write("{} :  Epoch {} ,accuracy : {:.4f}, time {:.0f}m {:.0f}s\n"
                .format(phase, stats.epochs[phase][i], stats.accuracies[phase][i], stats.times[phase][i] // 60, stats.times[phase][i] % 60))

    file.write('Training complete in {:.0f}m {:.0f}s \n'.format(
        stats.time_elapsed // 60, stats.time_elapsed % 60))
    file.write('Best val Acc: {:4f} \n\n'.format(stats.best_acc))
    file.close()

    lrs = []
    accs = []
    for s in lrstats:
        lrs.append(s[0])
        accs.append(s[1])

    plt.figure(frameon  = False)
    plt.plot(lrs,  accs)
    plt.xlabel('Learning Rate')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.savefig(folder +"n_epoch_{}__Resnet{}__batch_size_{}__trained_layers_{}__n_outputs_{}.pdf".format(n_epoch, resnet, batch_size, trained_layers, n_outputs))
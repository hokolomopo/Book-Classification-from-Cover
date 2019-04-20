from __future__ import print_function, division

import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np
import torchvision
from torchvision import datasets, models, transforms
import matplotlib.pyplot as plt
import time
import os
import copy

from stats import Statistics
from bookDataset import BookDataset
import cyclic_sceduler

def load_resnet(n):
    if n == 18:
        return models.resnet18(pretrained=True)
    elif n == 34:
        return models.resnet34(pretrained=True)
    elif n == 50:
        return models.resnet50(pretrained=True)
    elif n == 101:
        return models.resnet101(pretrained=True)
    elif n == 152:
        return models.resnet152(pretrained=True)

def change_model(model, trained_layers, n_outputs):
    """
    Freez the layers of the models expect the ones on top and add some layers on top of the mode
    """
    for param in model.parameters():
        param.requires_grad = False

    # Count the number of layers
    dpt = 0
    for child in model.children():
        dpt += 1

    # Unfreeze last trained_layers layers
    ct = 0
    for child in model.children():
        ct += 1
        if ct > dpt - (trained_layers - 1):
            for param in child.parameters():
                param.requires_grad = True

    num_ftrs = model.fc.in_features

    model.fc = nn.Sequential(
            nn.Linear(num_ftrs, 256), 
            nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(256, n_outputs),                   
            nn.LogSoftmax(dim=1))


    return model

def train_model(model, dataloaders, dataset_sizes, batch_size, criterion, optimizer, scheduler = None, num_epochs=25, device="cpu", scheduler_step="cycle"):
    """
    Train a model and return the trained model and statistics from the training
    """
    since = time.time()

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    stats = Statistics()

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                if scheduler and scheduler_step == "cycle":
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
                    print('Epoch {}, {:.2f}% time : {:.2f}'.format(epoch, progress, time.time() - start))
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
                        optimizer.step()
                        if(scheduler and scheduler_step == "batch"):
                            scheduler.batch_step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            end = time.time()


            stats.losses[phase].append(epoch_loss)
            stats.accuracies[phase].append(epoch_acc)
            stats.epochs[phase].append(epoch)
            stats.times[phase].append(end - start)

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

    return (model, stats)

if __name__ == "__main__":
    n_epoch = 30
    batch_size = 64
    n_workers = 4
    resnet = 18
    trained_layers = 10 
    n_outputs = 30

    finaLayer = "ReluDropoutSoftmax"
    filename = "batch64"

    min_lr = 1e-4
    max_lr = 6e-3

    lr = min_lr

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

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=min_lr, momentum=0.9)

    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler = cyclic_sceduler.CyclicLR(optimizer_ft, mode='triangular', base_lr=min_lr, max_lr=max_lr, step_size=2 * dataset_sizes['train'] / batch_size)

    model_ft, stats = train_model(model_ft, dataloaders, dataset_sizes, batch_size, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=n_epoch, device=device, scheduler_step="batch")

    folder = ""


    ######################################################################
    # Save results
    # ------------------
    file = open(folder + "{}.txt".format(filename), "a+")
    file.write("{}_Resnet{}, lr : {}, batch size : {}, trained_layers : {}, n_outputs : {}\n".format(finaLayer, resnet, lr, batch_size, trained_layers, n_outputs))

    for phase in ['train', 'val']:
        for i in range(len(stats.epochs[phase])):
            file.write("{} :  Epoch {} ,accuracy : {:.4f}, time {:.0f}m {:.0f}s\n"
                .format(phase, stats.epochs[phase][i], stats.accuracies[phase][i], stats.times[phase][i] // 60, stats.times[phase][i] % 60))

    file.write('Training complete in {:.0f}m {:.0f}s \n'.format(
        stats.time_elapsed // 60, stats.time_elapsed % 60))
    file.write('Best val Acc: {:4f} \n\n'.format(stats.best_acc))
    file.close()


    #Plot results
    plt.figure(frameon  = False)
    for x in ['train', 'val']:
        plt.plot(stats.epochs[x],  stats.accuracies[x], label=x)
    plt.xlabel('epoch')
    plt.ylabel('Accuracy')
    plt.grid(True)
    plt.legend()
    plt.savefig(folder +"graph/n_epoch_{}__Resnet{}__batch_size_{}__trained_layers_{}__n_outputs_{}__Accuracy.pdf".format(n_epoch, resnet, batch_size, trained_layers, n_outputs))

    plt.figure(frameon  = False)
    for x in ['train', 'val']:
        plt.plot(stats.epochs[x],  stats.losses[x], label=x)
    plt.xlabel('epoch')
    plt.ylabel('losses')
    plt.grid(True)
    plt.legend()
    plt.savefig(folder +"graph/n_epoch_{}__Resnet{}__batch_size_{}__trained_layers_{}__n_outputs_{}__Loss.pdf".format(n_epoch, resnet, batch_size, trained_layers, n_outputs))

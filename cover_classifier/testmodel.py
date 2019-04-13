"""
Transfer Learning Tutorial
==========================
**Author**: `Sasank Chilamkurthy <https://chsasank.github.io>`_
In this tutorial, you will learn how to train your network using
transfer learning. You can read more about the transfer learning at `cs231n
notes <https://cs231n.github.io/transfer-learning/>`__
Quoting these notes,
    In practice, very few people train an entire Convolutional Network
    from scratch (with random initialization), because it is relatively
    rare to have a dataset of sufficient size. Instead, it is common to
    pretrain a ConvNet on a very large dataset (e.g. ImageNet, which
    contains 1.2 million images with 1000 categories), and then use the
    ConvNet either as an initialization or a fixed feature extractor for
    the task of interest.
These two major transfer learning scenarios look as follows:
-  **Finetuning the convnet**: Instead of random initializaion, we
   initialize the network with a pretrained network, like the one that is
   trained on imagenet 1000 dataset. Rest of the training looks as
   usual.
-  **ConvNet as fixed feature extractor**: Here, we will freeze the weights
   for all of the network except that of the final fully connected
   layer. This last fully connected layer is replaced with a new one
   with random weights and only this layer is trained.
"""
# License: BSD
# Author: Sasank Chilamkurthy

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
import liboptim.cyclic_sceduler

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
    # model.fc = nn.Linear(num_ftrs, n_outputs)

    model.fc = nn.Sequential(
            # nn.Linear(num_ftrs, 256), 
            # nn.ReLU(), 
            nn.Dropout(0.4),
            nn.Linear(num_ftrs, n_outputs),                   
            nn.LogSoftmax(dim=1))


    return model


def imshow(inp, title=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(1)  # pause a bit so that plots are updated

######################################################################
# Visualizing the model predictions
# ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
#
# Generic function to display predictions for a few images
#
def visualize_model(model, num_images=6):
    was_training = model.training
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, (inputs, labels) in enumerate(dataloaders['val']):
            inputs = inputs.to(device)
            labels = labels.to(device)

            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for j in range(inputs.size()[0]):
                images_so_far += 1
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(class_names[preds[j]]))
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)
                    return
        model.train(mode=was_training)

def train_model(model, dataloaders, dataset_sizes, batch_size, criterion, optimizer, scheduler, num_epochs=25, device="cpu", scheduler_step="cycle"):
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
                if(scheduler_step == "cycle"):
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
                    print('Epoch {} : lr {:.5f}, {:.2f}% time : {:.2f}'.format(epoch, scheduler.get_lr()[0], progress, time.time() - start))
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
                        if(scheduler_step == "batch"):
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
    # plt.ion()   # interactive mode
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
    # min_lr = 5e-3


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
    # Visualize a few images
    # ^^^^^^^^^^^^^^^^^^^^^^
    # Let's visualize a few training images so as to understand the data
    # augmentations.


    # inputs, classes = next(iter(dataloaders['train']))


    # # Make a grid from batch
    # out = torchvision.utils.make_grid(inputs)

    # imshow(out, title=[class_names[x] for x in classes])


    ######################################################################
    # Training the model
    # ------------------
    model_ft = load_resnet(resnet)
    model_ft = change_model(model_ft, trained_layers, n_outputs)
    model_ft = model_ft.to(device)

    criterion = nn.CrossEntropyLoss()

    # Observe that all parameters are being optimized
    optimizer_ft = optim.SGD(model_ft.parameters(), lr=min_lr, momentum=0.9)

    # Decay LR by a factor of 0.1 every 7 epochs
    # exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)
    exp_lr_scheduler = liboptim.cyclic_sceduler.CyclicLR(optimizer_ft, mode='triangular', base_lr=min_lr, max_lr=max_lr, step_size=2 * dataset_sizes['train'] / batch_size)

    ######################################################################
    # Train and evaluate
    # ^^^^^^^^^^^^^^^^^^
    #
    # It should take around 15-25 min on CPU. On GPU though, it takes less than a
    # minute.
    #

    model_ft, stats = train_model(model_ft, dataloaders, dataset_sizes, batch_size, criterion, optimizer_ft, exp_lr_scheduler,
                        num_epochs=n_epoch, device=device, scheduler_step="batch")

    folder = ""

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

    # ######################################################################
    # #

    # visualize_model(model_ft)

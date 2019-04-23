from testmodel import *

if __name__ == "__main__":
    n_epoch = 25
    batch_size = 64
    n_workers = 2
    resnet = 18
    trained_layers = 10 
    n_outputs = 30

    min_lr = 1e-4
    max_lr = 6e-3


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

    folder = "cover_final/"

    optimizer_ft = optim.SGD(model_ft.parameters(), lr=min_lr, momentum=0.9)

    exp_lr_scheduler = cyclic_sceduler.CyclicLR(optimizer_ft, mode='triangular', base_lr=min_lr, max_lr=max_lr, step_size=2 * dataset_sizes['train'] / batch_size)

    model_ft, stats = train_model(model_ft, dataloaders, dataset_sizes, batch_size, criterion, optimizer_ft, None,
                        num_epochs=n_epoch, device=device, scheduler_step="batch")



    ######################################################################
    # Save results
    # ------------------


    # Save training stats into a file
    file = open(folder + "results.txt", "a+")
    file.write("Resnet{}, batch size : {}, trained_layers : {}, n_outputs : {}\n".format(resnet, batch_size, trained_layers, n_outputs))

    for phase in ['train', 'val']:
        for i in range(len(stats.epochs[phase])):
            file.write("{} :  Epoch {} ,accuracy : {:.4f}, time {:.0f}m {:.0f}s\n"
                .format(phase, stats.epochs[phase][i], stats.accuracies[phase][i], stats.times[phase][i] // 60, stats.times[phase][i] % 60))

    file.write('Training complete in {:.0f}m {:.0f}s \n'.format(
        stats.time_elapsed // 60, stats.time_elapsed % 60))
    file.write('Best val Acc: {:4f} \n\n\n'.format(stats.best_acc))
    file.close()

    for x in ['train', 'val']:
        plt.figure(frameon  = False)
        plt.plot(stats.epochs[x],  stats.accuracies[x])
        plt.xlabel('Epochs')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.savefig(folder +"n_epoch_{}__Resnet{}__batch_size_{}__trained_layers_{}__n_outputs_{}__Loss_{}.pdf".format(n_epoch, resnet, batch_size, trained_layers, n_outputs,x))


    # Save model
    torch.save(model_ft.state_dict(), folder + "model{}".format(batch_size))

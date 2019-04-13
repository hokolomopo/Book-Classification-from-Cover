from testmodel import *

if __name__ == "__main__":
    # plt.ion()   # interactive mode
    n_epoch = 10
    batch_size = 8
    n_workers = 2
    trained_layers = 10 
    n_outputs = 30
    lr = 1e-4

    resnets = [18, 34, 50, 101]

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

    criterion = nn.CrossEntropyLoss()

    stats_list = []
    folder = "resnet_comp/"

    for resnet in resnets:
        print('\nLearnig rate {}'.format(lr))
        print('-' * 10)

        model_ft = load_resnet(resnet)
        model_ft = change_model(model_ft, trained_layers, n_outputs)
        model_ft = model_ft.to(device)

        # Observe that all parameters are being optimized
        optimizer_ft = optim.SGD(model_ft.parameters(), lr=lr, momentum=0.9)

        # Decay LR by a factor of 0.1 every 7 epochs
        exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

        model_ft, stats = train_model(model_ft, dataloaders, dataset_sizes, batch_size, criterion, optimizer_ft, exp_lr_scheduler,
                            num_epochs=n_epoch, device=device)

        stats_list.append(stats)


        file = open(folder + "resnet_comp.txt", "a+")
        file.write("Resnet{}, lr : {}, batch size : {}, trained_layers : {}, n_outputs : {}\n".format(resnet, lr, batch_size, trained_layers, n_outputs))

        for phase in ['train', 'val']:
            for i in range(len(stats.epochs[phase])):
                file.write("{} :  Epoch {} ,accuracy : {:.4f}, time {:.0f}m {:.0f}s\n"
                    .format(phase, stats.epochs[phase][i], stats.accuracies[phase][i], stats.times[phase][i] // 60, stats.times[phase][i] % 60))

        file.write('Training complete in {:.0f}m {:.0f}s \n'.format(
            stats.time_elapsed // 60, stats.time_elapsed % 60))
        file.write('Best val Acc: {:4f} \n\n'.format(stats.best_acc))
        file.close()


    for x in ['train', 'val']:
        plt.figure(frameon  = False)
        for i in range(len(lrs)):
            plt.plot(stats_list[i].epochs[x],  stats_list[i].accuracies[x], label="lr {}".format(lrs[i]))
        plt.xlabel('epoch')
        plt.ylabel('Accuracy')
        plt.grid(True)
        plt.legend()
        plt.savefig(folder +"n_epoch_{}__Resnet{}__batch_size_{}__trained_layers_{}__n_outputs_{}__Accuracy_{}.pdf".format(n_epoch, resnet, batch_size, trained_layers, n_outputs, x))

        plt.figure(frameon  = False)
        for i in range(len(lrs)):
            plt.plot(stats_list[i].epochs[x],  stats_list[i].losses[x], label="lr {}".format(lrs[i]))
        plt.xlabel('epoch')
        plt.ylabel('losses')
        plt.grid(True)
        plt.legend()
        plt.savefig(folder +"n_epoch_{}__Resnet{}__batch_size_{}__trained_layers_{}__n_outputs_{}__Loss_{}.pdf".format(n_epoch, resnet, batch_size, trained_layers, n_outputs,x))

    file = open(folder + "resnet_comp.txt", "a+")
    file.write('\n\n')
    file.close()

    # ######################################################################
    # #

    # visualize_model(model_ft)

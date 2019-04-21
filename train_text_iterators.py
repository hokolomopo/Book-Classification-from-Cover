from testmodel import *

def train_model(model, iterators, dataset_sizes, batch_size, criterion, optimizer, scheduler = None, num_epochs=25, device="cpu", scheduler_step="cycle"):
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

            for batch in iterators[phase]:
                inputs = batch.title
                labels = batch.label
                
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
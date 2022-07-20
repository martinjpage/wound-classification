from utils import config

import copy
import torch
from torch.optim.lr_scheduler import OneCycleLR


def create_scheduler(trainloader, optimiser):
    STEPS_PER_EPOCH = len(trainloader)
    TOTAL_STEPS = config.EPOCHS * STEPS_PER_EPOCH
    MAX_LRS = [p['lr'] for p in optimiser.param_groups]
    return OneCycleLR(optimiser, max_lr=MAX_LRS, total_steps=TOTAL_STEPS)


def train(train_loader, model, optimiser, criterion, scheduler, device):
    epoch_loss = 0
    epoch_acc = 0

    model.train() # Set model to training mode

    # Iterate over data
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)

        # zero the parameter gradients
        optimiser.zero_grad()

        # forward
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        y_pred, _ = model(x)

        loss = criterion(y_pred, y)

        acc_1, acc_5 = calculate_topk_accuracy(y_pred, y)

        loss.backward()

        optimiser.step()

        scheduler.step()

        epoch_loss += loss.item()
        epoch_acc_1 += acc_1.item()
        epoch_acc_5 += acc_5.item()

    epoch_loss /= len(dataloaders)
    epoch_acc_1 /= len(dataloaders)
    epoch_acc_5 /= len(dataloaders)

    return epoch_loss, epoch_acc_1, epoch_acc_5


def fit_model(dataloaders, model, criterion, optimiser, scheduler, device, epochs):
    since = time.time()

    model = model.to(device)
    criterion = criterion.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch}/{num_epochs - 1}')
        print('-' * 10)

    # Each epoch has a training and validation phase
    for phase in ['train', 'val']:
        if phase == 'train':
            model.train()  # Set model to training mode
        else:
            model.eval()  # Set model to evaluate mode

        running_loss = 0.0
        running_corrects = 0

        # Iterate over data.
        for inputs, labels in dataloaders[phase]:
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            optimiser.zero_grad()

            # forward
            # track history if only in train
            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)
                _, preds = torch.max(outputs, 1)
                loss = criterion(outputs, labels)

                # backward + optimize only if in training phase
                if phase == 'train':
                    loss.backward()
                    optimiser.step()

            # statistics
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
        if phase == 'train':
            scheduler.step()

        epoch_loss = running_loss / len(dataloaders[phase])
        epoch_acc = running_corrects.double() / len(dataloaders[phase])

        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

        # deep copy the model
        if phase == 'val' and epoch_acc > best_acc:
            best_acc = epoch_acc
            best_model_wts = copy.deepcopy(model.state_dict())

    print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best val Acc: {best_acc:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

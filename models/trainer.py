from utils import config

import time
import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR


def create_scheduler(trainloader, optimiser):
    STEPS_PER_EPOCH = len(trainloader)
    TOTAL_STEPS = config.EPOCHS * STEPS_PER_EPOCH
    MAX_LRS = [p['lr'] for p in optimiser.param_groups]
    return OneCycleLR(optimiser, max_lr=MAX_LRS, total_steps=TOTAL_STEPS)


def calculate_accuracy(y_preds, y):
    with torch.no_grad():
        batch_size = y.shape[0]
        _, predicted_class = torch.max(y_preds, 1)
        correct = predicted_class.eq(y.view(1, -1).expand_as(predicted_class))
        return correct/batch_size


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def fit_model(dataloaders, model, criterion, optimiser, scheduler, device, epochs):
    since = time.time()

    model = model.to(device)
    criterion = criterion.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(epochs):
        print(f'Epoch {epoch+1}/{epochs}')
        print('-' * 10)
        start_time = time.monotonic()

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
                    y_preds, _ = model(inputs)
                    _, pred_classes = torch.max(y_preds, 1)
                    loss = criterion(y_preds, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(pred_classes == labels.data)
            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} - {phase} | Epoch Time: {epoch_mins}m {epoch_secs}s')
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


def get_predictions(model, dataloader, device):

    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():

        for x, y in dataloader:

            x = x.to(device)

            y_pred, _ = model(x)

            y_prob = F.softmax(y_pred, dim = -1)
            top_pred = y_prob.argmax(1, keepdim = True)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim = 0)
    labels = torch.cat(labels, dim = 0)
    probs = torch.cat(probs, dim = 0)

    return images, labels, probs


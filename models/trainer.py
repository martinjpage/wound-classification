import numpy as np

from utils import config

import time
import copy
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import ConfusionMatrixDisplay

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


def fit_model(dataloaders, model, criterion, optimiser, scheduler, device, epochs, patience):
    since = time.time()

    model = model.to(device)
    criterion = criterion.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_fscore = 0.0
    last_loss = 100
    patience = patience
    trigger_times = 0

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
            true_labels = torch.empty(0)
            predicted_labels = torch.empty(0)

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
                true_labels = torch.cat((true_labels, labels.data))
                predicted_labels = torch.cat((predicted_labels, pred_classes.data))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} - {phase} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            acc, gmean, fscore, j_index = calculate_metrics(true_labels, predicted_labels)
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f} F-score: {fscore:.4f}  '
                  f'G Score: {gmean:.4f} J Score: {j_index:.4f}')
            # plot_confusion_matrix(true_labels, predicted_labels, classes=['False', 'True'])

            # deep copy the model
            if phase == 'val' and fscore > best_fscore and gmean > 0.5:
                best_fscore = fscore
                best_model_wts = copy.deepcopy(model.state_dict())

            # early stopping
            if phase == 'val' and epoch_loss > last_loss:
                trigger_times += 1
                last_loss = epoch_loss
                print('Trigger Times:', trigger_times)

                if trigger_times >= patience:
                    print('Early stopping.')

                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best F-score: {best_fscore:4f}')

                    model.load_state_dict(best_model_wts)
                    return model

            elif phase == 'val' and epoch_loss <= last_loss:
                print('trigger times: 0')
                trigger_times = 0
                last_loss = epoch_loss

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best F-score: {best_fscore:4f}')

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
            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    pred_labels = torch.argmax(probs, 1)

    return images, labels, pred_labels


def calculate_metrics(labels, pred_labels):
    accuracy = torch.sum(pred_labels == labels)/len(pred_labels)
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    sensitivity = tp / (tp + fn)  # aka recall
    specificity = tn / (fp + tn)
    g_mean = np.sqrt(sensitivity*specificity)  # Fowlkes–Mallows index
    precision = tp / (tp + fp)
    f_score = (2 * precision * sensitivity) / (precision + sensitivity)
    jaccard_index = tp / (tp+fp+fn)
    return accuracy, g_mean, f_score, jaccard_index


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    acc, gmean, fscore, j_index = calculate_metrics(labels, pred_labels)
    plt.title(f'Accuracy: {acc:.2f}, G-mean: {gmean:.2f}, F Score: {fscore:.2f}')
    plt.xticks(rotation=20)
    plt.show()

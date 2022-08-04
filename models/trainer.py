import numpy as np

from utils import config
from utils.utils import calculate_binary_metrics, calculate_multi_metrics

import wandb
import time
import copy
import torch
from torch.optim.lr_scheduler import OneCycleLR
import torch.nn as nn
import torch.nn.functional as F


def create_scheduler(trainloader, optimiser):
    STEPS_PER_EPOCH = len(trainloader)
    TOTAL_STEPS = config.EPOCHS * STEPS_PER_EPOCH
    MAX_LRS = [p['lr'] for p in optimiser.param_groups]
    return OneCycleLR(optimiser, max_lr=MAX_LRS, total_steps=TOTAL_STEPS)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


def fit_model(dataloaders, model, criterion, optimiser, scheduler, device, epochs, selection_metric,
              early_stop=True, stop_metric='loss', patience=3):
    since = time.time()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_selection_score = 0.0
    last_early_metric = -np.inf
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

                    # backward + optimise only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                true_labels = torch.cat((true_labels, labels.data))
                predicted_labels = torch.cat((predicted_labels, pred_classes.data))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metrics = calculate_binary_metrics(true_labels, predicted_labels, as_dict=True)
            epoch_metrics['loss'] = epoch_loss

            wandb.log({f'{phase}_loss': epoch_metrics['loss'],
                       f'{phase}_acc': epoch_metrics['acc'],
                       f'{phase}_fscore': epoch_metrics['fscore'],
                       f'{phase}_gmean': epoch_metrics['gmean'],
                       f'{phase}_jindex': epoch_metrics['jindex'],
                       })
            wandb.watch(model)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} - {phase} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'{phase} Loss: {epoch_metrics["loss"]:.4f} Acc: {epoch_metrics["acc"]:.4f} '
                  f'F-score: {epoch_metrics["fscore"]:.4f} G Score: {epoch_metrics["gmean"]:.4f} '
                  f'J Score: {epoch_metrics["jindex"]:.4f}')

            # deep copy the best model
            if phase == 'val' and epoch_metrics[selection_metric] > best_selection_score and \
                    epoch_metrics["gmean"] > 0.5:
                best_selection_score = epoch_metrics[selection_metric]
                best_model_wts = copy.deepcopy(model.state_dict())

            # early stopping
            if early_stop and phase == 'val' and epoch_metrics[stop_metric] < last_early_metric:
                trigger_times += 1
                last_early_metric = epoch_metrics[stop_metric]
                print(f'Easy Stop Trigger Times:{trigger_times} of {patience}.')

                if trigger_times >= patience:
                    print(f'Early stopping. Trigger threshold reached on {stop_metric}.')

                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best {selection_metric}: {best_selection_score:4f}')

                    model.load_state_dict(best_model_wts)
                    return model

            elif early_stop and phase == 'val' and epoch_metrics[stop_metric] >= last_early_metric:
                print('Resetting early stopping trigger to 0.')
                trigger_times = 0
                last_early_metric = epoch_metrics[stop_metric]

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best {selection_metric}: {best_selection_score:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        all_probs = F.softmax(inputs, dim=-1)
        class_probs, y_pred = torch.max(all_probs, dim=1)
        bce_loss = F.binary_cross_entropy(class_probs,  targets.float())
        loss = self.alpha * (1 - torch.exp(-bce_loss)) ** self.gamma * bce_loss
        return loss


def fit_multi_model(dataloaders, model, criterion, optimiser, scheduler, device, epochs,
              selection_metric, early_stop=True, stop_metric='loss', patience=3):
    since = time.time()

    model = model.to(device)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_selection_score = 0.0
    last_early_metric = -np.inf
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

                    # backward + optimise only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimiser.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                true_labels = torch.cat((true_labels, labels.data))
                predicted_labels = torch.cat((predicted_labels, pred_classes.data))

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metrics = calculate_multi_metrics(true_labels, predicted_labels, as_dict=True)
            epoch_metrics['loss'] = epoch_loss

            wandb.log({f'{phase}_loss': epoch_metrics['loss'],
                       f'{phase}_acc': epoch_metrics['acc'],
                       f'{phase}_fscore': epoch_metrics['fscore']
                       })
            wandb.watch(model)

            end_time = time.monotonic()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            print(f'Epoch: {epoch + 1:02} - {phase} | Epoch Time: {epoch_mins}m {epoch_secs}s')
            print(f'{phase} Loss: {epoch_metrics["loss"]:.4f} Acc: {epoch_metrics["acc"]:.4f} '
                  f'F-score: {epoch_metrics["fscore"]:.4f}')

            # deep copy the best model
            if phase == 'val' and epoch_metrics[selection_metric] > best_selection_score:
                best_selection_score = epoch_metrics[selection_metric]
                best_model_wts = copy.deepcopy(model.state_dict())

            # early stopping
            if early_stop and phase == 'val' and epoch_metrics[stop_metric] < last_early_metric:
                trigger_times += 1
                last_early_metric = epoch_metrics[stop_metric]
                print(f'Easy Stop Trigger Times:{trigger_times} of {patience}.')

                if trigger_times >= patience:
                    print(f'Early stopping. Trigger threshold reached on {stop_metric}.')

                    time_elapsed = time.time() - since
                    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
                    print(f'Best {selection_metric}: {best_selection_score:4f}')

                    model.load_state_dict(best_model_wts)
                    return model

            elif early_stop and phase == 'val' and epoch_metrics[stop_metric] >= last_early_metric:
                print('Resetting early stopping trigger to 0.')
                trigger_times = 0
                last_early_metric = epoch_metrics[stop_metric]

        print()

    time_elapsed = time.time() - since
    print(f'Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s')
    print(f'Best {selection_metric}: {best_selection_score:4f}')

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model

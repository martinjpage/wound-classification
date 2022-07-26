from utils import utils
from utils import config
from models.trainer import create_scheduler, fit_model, get_predictions, plot_confusion_matrix
from models.resnet50 import create_resnet50
from models.lr_finder import LRFinder
from data.dataloader import create_dataloader
from data.data_transformer import train_transformer, val_transformer, test_transformer

import os
import torch
import torch.nn as nn
from torch import optim

# Setup CPU/GPU
device = utils.setup_gpu()

# Data Paths
image_directory = os.path.join(os. getcwd(), 'data', 'images', 'res300')
train_file = os.path.join(os. getcwd(), 'data', 'data_split_clot', 'train_set.csv')
validation_file = os.path.join(os. getcwd(), 'data', 'data_split_clot', 'val_set.csv')
test_file = os.path.join(os. getcwd(), 'data', 'data_split_clot', 'test_set.csv')


# Data Loading
# ToDo: tiling?; transforms
trainloader = create_dataloader(images_csv=train_file, image_dir=image_directory, transform=train_transformer(),
                                batch_size=config.BATCH_SIZE, shuffle=True)
valloader = create_dataloader(images_csv=validation_file, image_dir=image_directory, transform=val_transformer(),
                              batch_size=config.BATCH_SIZE, shuffle=True)
testloader = create_dataloader(images_csv=test_file, image_dir=image_directory, transform=test_transformer())
dataloaders = {"train": trainloader, "val": valloader, "test": testloader}


# LR Finding
model = create_resnet50().to(device)
criterion = nn.CrossEntropyLoss().to(device)

# optimiser = optim.AdamW(model.parameters(), lr=config.START_LR, weight_decay=0)
# lr_finder = LRFinder(model, optimiser, criterion, device)
# lr_finder.range_test(trainloader, config.END_LR, config.NUM_ITER)
# lr_finder.plot_lr_finder(skip_start=25, skip_end=10)

# Training
# params = [
#           {'params': model.conv1.parameters(), 'lr': config.FOUND_LR / 10},
#           {'params': model.bn1.parameters(), 'lr': config.FOUND_LR / 10},
#           {'params': model.layer1.parameters(), 'lr': config.FOUND_LR / 8},
#           {'params': model.layer2.parameters(), 'lr': config.FOUND_LR / 6},
#           {'params': model.layer3.parameters(), 'lr': config.FOUND_LR / 4},
#           {'params': model.layer4.parameters(), 'lr': config.FOUND_LR / 2},
#           {'params': model.fc.parameters()}
#          ]
# ToDo: weight decay if use different learning rates
# ToDo: better measurement than accuracy; fewer transformations on training; tiling; up/down-sampling
optimiser = optim.AdamW(model.parameters(), lr=config.FOUND_LR, weight_decay=config.WEIGHT_DECAY)
scheduler = create_scheduler(trainloader=trainloader, optimiser=optimiser)
model = fit_model(dataloaders, model, criterion, optimiser, scheduler, device, epochs=config.EPOCHS,
                  selection_metric='fscore', early_stop=True, stop_metric='fscore', patience=4)


# Model Save
model_name = f'resnet50_lr{config.FOUND_LR}-weight{config.WEIGHT_DECAY}-fscore-30epoch.pt'
torch.save(model.state_dict(), model_name)


# Testing
images, labels, pred_labels = get_predictions(model, valloader, device)
plot_confusion_matrix(labels, pred_labels, classes=['False', 'True'])


# ToDo: train function; hyperparameter tuning (LR Finder); model;
#  keeping track of results/plotting loss/accuracy



# if __name__ == '__main__':
#     # view images in dataloader
#     for train_features, train_labels in dataloader["train"]:
#         print(f"Feature batch shape: {train_features.size()}")
#         print(f"Labels batch shape: {train_labels.size()}")
#         img = train_features[0]
#         label = train_labels.item()
#         utils.show_image(img, label, normalise=True)

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
import wandb


# ToDo: downsample and dilation in CNN
# Configure Experiment and Logging
project_name = "my-test-project"
experiment_name = 'resnet50'
run_config = {
    "learning_rate": config.FOUND_LR,
    "epochs": config.EPOCHS,
    "batch_size": config.BATCH_SIZE,
    "weight_decay": config.WEIGHT_DECAY,
    "loss": "cross_entropy",
    "selection_metric": config.SELECTION_METRIC,
    "stop_metric": config.STOP_METRIC,
    "architecture": config.ARCHITECTURE
}

WANDB_KEY = '5daa291a45f220cbec42e63995626bb6c5712839'
wandb.login(key=WANDB_KEY)
wandb.init(project=project_name, name=experiment_name, config=run_config)

# Setup CPU/GPU
device = utils.setup_gpu()

# Data Paths
image_directory = os.path.join(os. getcwd(), 'data', 'images', 'res300')
train_file = os.path.join(os. getcwd(), 'data', 'data_split_clot', 'train_set.csv')
validation_file = os.path.join(os. getcwd(), 'data', 'data_split_clot', 'val_set.csv')
test_file = os.path.join(os. getcwd(), 'data', 'data_split_clot', 'test_set.csv')


# Data Loading
# ToDo: tiling?; transforms
trainloader = create_dataloader(images_csv=train_file, image_dir=image_directory,
                                transform=train_transformer(config.ARCHITECTURE),
                                batch_size=config.BATCH_SIZE, shuffle=True)
valloader = create_dataloader(images_csv=validation_file, image_dir=image_directory,
                              transform=val_transformer(config.ARCHITECTURE),
                              batch_size=config.BATCH_SIZE, shuffle=True)
testloader = create_dataloader(images_csv=test_file, image_dir=image_directory,
                               transform=test_transformer(config.ARCHITECTURE))
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
wandb.finish()

# Model Save
model_name = f'resnet50_lr{config.FOUND_LR}-weight{config.WEIGHT_DECAY}-fscore-30epoch-wandb.pt'
torch.save(model.state_dict(), model_name)


# Testing
images, labels, pred_labels = get_predictions(model, valloader, device)
plot_confusion_matrix(labels, pred_labels, classes=['False', 'True'])

from utils import utils
from utils import config
from models.trainer import create_scheduler, fit_model
from models.tester import get_predictions, plot_confusion_matrix
from models.resnet50 import create_resnet50
from models.resnet101 import create_resnet101
from models.lr_finder import LRFinder
from data.dataloader import create_dataloader
from data.data_transformer import train_transformer, val_transformer, test_transformer

import os
import torch
import torch.nn as nn
from torchvision.ops import sigmoid_focal_loss
from torch import optim
import wandb


# Configure Experiment and Logging
project_name = "wound_clot_classification"
experiment_name = 'resnet101_fscore_ce_lrs'
run_config = {
    "learning_rate": config.FOUND_LR,
    "epochs": config.EPOCHS,
    "batch_size": config.BATCH_SIZE,
    "weight_decay": config.WEIGHT_DECAY,
    "loss": "cross_entropy",
    "selection_metric": config.SELECTION_METRIC,
    "early_stop": config.EARYSTOP,
    "stop_metric": config.STOP_METRIC,
    "patience": config.PATIENCE,
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
trainloader = create_dataloader(images_csv=train_file, image_dir=image_directory, target=config.CSV_CLOT_COLUMN,
                                transform=train_transformer(config.ARCHITECTURE),
                                batch_size=config.BATCH_SIZE, shuffle=True)
valloader = create_dataloader(images_csv=validation_file, image_dir=image_directory, target=config.CSV_CLOT_COLUMN,
                              transform=val_transformer(config.ARCHITECTURE),
                              batch_size=config.BATCH_SIZE, shuffle=True)
testloader = create_dataloader(images_csv=test_file, image_dir=image_directory, target=config.CSV_CLOT_COLUMN,
                               transform=test_transformer(config.ARCHITECTURE))
dataloaders = {"train": trainloader, "val": valloader, "test": testloader}


# Training
model = None
if config.ARCHITECTURE == "resnet50":
    model = create_resnet50().to(device)
elif config.ARCHITECTURE == "resnet101":
    model = create_resnet101().to(device)
else:
    raise ValueError(f"No model {config.ARCHITECTURE}.")

criterion = nn.CrossEntropyLoss()

# LR Finding
# optimiser = optim.AdamW(model.parameters(), lr=config.START_LR, weight_decay=0)
# lr_finder = LRFinder(model, optimiser, criterion, device)
# lr_finder.range_test(trainloader, config.END_LR, config.NUM_ITER)
# lr_finder.plot_lr_finder(skip_start=25, skip_end=10)

# Training
params = [
          {'params': model.conv1.parameters(), 'lr': config.FOUND_LR / 10},
          {'params': model.bn1.parameters(), 'lr': config.FOUND_LR / 10},
          {'params': model.layer1.parameters(), 'lr': config.FOUND_LR / 8},
          {'params': model.layer2.parameters(), 'lr': config.FOUND_LR / 6},
          {'params': model.layer3.parameters(), 'lr': config.FOUND_LR / 4},
          {'params': model.layer4.parameters(), 'lr': config.FOUND_LR / 2},
          {'params': model.fc.parameters()}
         ]
# ToDo: weight decay if use different learning rates?
# ToDo: tiling; up/down-sampling; criterion = focal loss
# optimiser = optim.AdamW(model.parameters(), lr=config.FOUND_LR, weight_decay=config.WEIGHT_DECAY)
optimiser = optim.AdamW(params, lr=config.FOUND_LR, weight_decay=config.WEIGHT_DECAY)
scheduler = create_scheduler(trainloader=trainloader, optimiser=optimiser)
model = fit_model(dataloaders, model, criterion, optimiser, scheduler, device, epochs=config.EPOCHS,
                  selection_metric=config.SELECTION_METRIC, early_stop=config.EARYSTOP, stop_metric=config.STOP_METRIC,
                  patience=config.PATIENCE)
wandb.finish()

# Model Save
model_name = f'{experiment_name}-epoch-{config.EPOCHS}-{config.SELECTION_METRIC}.pt'
torch.save(model.state_dict(), model_name)


# Testing
images, labels, probs, pred_labels = get_predictions(model, testloader, device)
plot_confusion_matrix(labels, pred_labels, classes=['False', 'True'])

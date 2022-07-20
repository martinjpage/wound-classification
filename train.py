from utils import utils
from utils import config
from models.trainer import create_scheduler
from models.resnet50 import create_resnet50
from data.dataloader import create_dataloader
from data.data_transformer import train_transformer, val_transformer, test_transformer

import os
import torch.nn as nn
from torch import optim

# Setup CPU/GPU
device = utils.setup_gpu()

# Data Paths
image_directory = os.path.join(os. getcwd(), 'data', 'images')
train_file = os.path.join(os. getcwd(), 'data', 'train_set.csv')
validation_file = os.path.join(os. getcwd(), 'data', 'val_set.csv')
test_file = os.path.join(os. getcwd(), 'data', 'test_set.csv')


# Data Loading
# ToDo: tiling?
trainloader = create_dataloader(images_csv=train_file, image_dir=image_directory, transform=train_transformer(),
                                batch_size=config.BATCH_SIZE, shuffle=True)
valloader = create_dataloader(images_csv=validation_file, image_dir=image_directory, transform=val_transformer(),
                              batch_size=config.BATCH_SIZE, shuffle=True)
testloader = create_dataloader(images_csv=test_file, image_dir=image_directory, transform=test_transformer())
dataloader = {"train": trainloader, "val": valloader, "test": testloader}


# LR Finding
model = create_resnet50().to(device)
criterion = nn.CrossEntropyLoss().to(device)

# optimiser = optim.AdamW(model.parameters(), lr=config.START_LR, weight_decay=0)
# lr_finder = LRFinder(model, optimiser, criterion, device)
# lr_finder.range_test(trainloader, config.END_LR, config.NUM_ITER)
# lr_finder.plot_lr_finder()

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
optimiser = optim.AdamW(model.parameters(), lr=config.FOUND_LR, weight_decay=config.WEIGHT_DECAY)
scheduler = create_scheduler(trainloader=trainloader, optimiser=optimiser)


# ToDo: train function; hyperparameter tuning (LR Finder); model;
#  keeping track of results/plotting loss/accuracy




if __name__ == '__main__':
    # view images in dataloader
    for train_features, train_labels in dataloader["train"]:
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0]
        label = train_labels.item()
        utils.show_image(img, label, normalise=True)

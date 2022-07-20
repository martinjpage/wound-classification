from utils import utils
from utils import config
from data.dataloader import create_dataloader
from data.data_transformer import train_transformer, val_transformer, test_transformer

import os
import torch
import torch.nn as nn


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


# Training
criterion = nn.CrossEntropyLoss()
# ToDo: confirm optimiser & scheduler
optimiser = None
scheduler = None

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

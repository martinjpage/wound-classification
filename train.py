from utils import utils
from utils import config
from data.dataloader import ClassificationDataset
from data.data_transformer import transform_data

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.utils.data as data


# Setup
device = utils.setup_gpu()

# Data Paths
image_directory = os.path.join(os. getcwd(), 'data', 'images')
csv_file = os.path.join(os. getcwd(), 'data', 'filenames.csv')
# ToDo: train/validation split --> data.random_split(); test split


# Data Transformations
composed = transform_data()
# ToDo: which augementation for training (but not validation set?) - adds images?;
#  calculate mean/std on dataset for normalisation OR pretrained means OR same size and norm as pretrained?

# Data Loading
dataset = ClassificationDataset(images_csv=csv_file, image_dir=image_directory, transform=composed)
trainloader = DataLoader(dataset=dataset, batch_size=config.BATCH_SIZE, shuffle=True)
# ToDo: validation loader and combine in dict

# Training
criterion = nn.CrossEntropyLoss()
# ToDo: confirm optimiser & scheduler
optimiser = None
scheduler = None

# ToDo: train function; hyperparameter tuning (LR Finder); model;
#  keeping track of results/plotting loss/accuracy


if __name__ == '__main__':
    # view images in trainloader
    for train_features, train_labels in trainloader:
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        img = train_features[0]
        label = train_labels.item()
        utils.show_image(img, label)

from utils import config
from utils import utils
from models.resnet50 import create_resnet50
from models.resnet101 import create_resnet101
from models.tester import get_predictions, plot_confusion_matrix, plot_most_incorrect
from data.dataloader import create_dataloader
from data.data_transformer import train_transformer, val_transformer, test_transformer

import torch
import os


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


# Load Model
model_path = os.path.join(os. getcwd(), 'reports', 'saved_models', 'resnet50_gmean_ce-lr0.005-epoch-50.pt')

model = None
if config.ARCHITECTURE == "resnet50":
    model = create_resnet50().to(device)
elif config.ARCHITECTURE == "resnet101":
    model = create_resnet101().to(device)
else:
    raise ValueError(f"No model {config.ARCHITECTURE}.")

model.load_state_dict(torch.load(model_path))
model.eval()

# Testing
images, labels, probs, pred_labels = get_predictions(model, valloader, device)
plot_confusion_matrix(labels, pred_labels, classes=['False', 'True'])

images, labels, probs, pred_labels = get_predictions(model, testloader, device)
plot_confusion_matrix(labels, pred_labels, classes=['False', 'True'])

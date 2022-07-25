from utils import utils
from utils import config
from models.trainer import create_scheduler, fit_model, get_predictions
from models.resnet50 import create_resnet50
from data.dataloader import create_dataloader
from data.data_transformer import train_transformer, val_transformer, test_transformer

import os
import torch
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
dataloaders = {"train": trainloader, "val": valloader, "test": testloader}


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
# ToDo: weight decay if use different learning rates
# ToDo: better measurement than accuracy; fewer transformations on training; up/down-sampling
# ToDo: stop training at plateau
optimiser = optim.AdamW(model.parameters(), lr=config.FOUND_LR, weight_decay=0)
scheduler = create_scheduler(trainloader=trainloader, optimiser=optimiser)
model_fit = fit_model(dataloaders, model, criterion, optimiser, scheduler, device, epochs=config.EPOCHS)

torch.save(model.state_dict(), f'resnet50_lr{config.FOUND_LR}.pt')

images, labels, probs = get_predictions(model, testloader, device)
pred_labels = torch.argmax(probs, 1)


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

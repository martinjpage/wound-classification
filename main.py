from data.dataloader import create_dataloader
from data.data_transformer import train_transformer, val_transformer, test_transformer
from utils import config

import os
from utils import utils

# Project Configuration
image_directory = os.path.join(os. getcwd(), 'data', 'images', 'res300')
# csv_file = os.path.join(os. getcwd(), 'data', 'all_image_filenames.csv')

# Create CSV of image file names (after export, add labels to CSV manually)
# utils.get_image_filenames(image_directory, csv_file)

# df = utils.load_image_filenames(csv_file)
# utils.change_image_file_resolutions(csv_file, "res600", "res300")
# Split CSV of image file names with labels in test, train, validation sets and export new CSVs
# utils.create_data_split(csv_file, train_size=0.70, valid_size=0.20, test_size=0.10)

# from data import dataloader as dl
# from data.data_transformer import transform_data
#
# cds = dl.ClassificationDataset(images_csv=csv_file, image_dir=image_directory, transform=transform_data())

# Data Paths
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


if __name__ == '__main__':
    # view images in dataloader
    dataloader = dataloaders["train"]
    for train_features, train_labels in dataloader:
        print(f"Feature batch shape: {train_features.size()}")
        print(f"Labels batch shape: {train_labels.size()}")
        for i in range(dataloader.batch_size):
            img = train_features[i]
            label = train_labels[i].item()
            utils.show_image(img, label, normalise=True)

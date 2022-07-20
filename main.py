import os
from utils import utils

# Project Configuration
image_directory = os.path.join(os. getcwd(), 'data', 'images')
csv_file = os.path.join(os. getcwd(), 'data', 'all_image_filenames.csv')

# Create CSV of image file names (after export, add labels to CSV manually)
# utils.get_image_filenames(image_directory, csv_file)

# df = utils.load_image_filenames(csv_file)

# Split CSV of image file names with labels in test, train, validation sets and export new CSVs
utils.create_data_split(csv_file)

# from data import dataloader as dl
# from data.data_transformer import transform_data
#
# cds = dl.ClassificationDataset(images_csv=csv_file, image_dir=image_directory, transform=transform_data())

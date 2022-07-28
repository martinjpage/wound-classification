from utils import utils
from utils import config as const

import os
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler


class ClassificationDataset(Dataset):
    """
    :param images_csv: path to CSV file with image filenames and categories (named accoriding to const)
    :param image_dir: path to folder with the actual images matching the names in images_csv
    :return Dataset class for preparing inputs for DataLoader
    """

    def __init__(self, images_csv: str, image_dir: str, target: str, transform=None):
        # image directory
        self.image_dir = image_dir
        # image paths and lables
        self.image_data = utils.load_image_filenames(images_csv, col=target)
        # transform to be used on image
        self.transform = transform
        self.target = target

    def __len__(self):
        # number of images in dataset
        return len(self.image_data)

    def __getitem__(self, idx):
        # image file path
        img_name = os.path.join(self.image_dir, self.image_data.iloc[idx][const.CSV_FILENAME_COLUMN])
        # open image file; convert to RGB to remove possible 4th alpha channel
        img = Image.open(img_name).convert('RGB')

        # class label for the image
        y = self.image_data.iloc[idx][self.target]
        # convert boolean into integer classes False = 0; True = 1
        y = int(y)

        # if there is any transform method, apply it onto the image
        if self.transform:
            img = self.transform(img)
        return img, y


def create_dataloader(images_csv: str, image_dir: str, target: str, transform=None, batch_size=1,
                      shuffle=False, over_sample=False):
    if over_sample:
        sampler = create_oversampler(images_csv, target)
    else:
        sampler = None
    dataset = ClassificationDataset(images_csv=images_csv, image_dir=image_dir, target=target, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=shuffle, sampler=sampler)
    return dataloader


def create_oversampler(images_csv, target):
    df = utils.load_image_filenames(images_csv, col=target)
    unique_classes, class_counts = np.unique(df[target], return_counts=True)
    class_weights = [sum(class_counts) / c for c in class_counts]
    weights = [class_weights[e] for e in df[target]]
    sampler = WeightedRandomSampler(weights, len(df[target]))
    return sampler

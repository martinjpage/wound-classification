from utils import config as const

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_image_filenames(image_directory, csv_output):
    image_files = os.listdir(image_directory)
    df = pd.DataFrame(image_files, columns=[const.CSV_FILENAME_COLUMN])
    df[const.CSV_CLOT_COLUMN] = ""
    df.to_csv(csv_output, index=False)
    print("Image file names written to file")


def load_image_filenames(path):
    return pd.read_csv(path, usecols=[const.CSV_FILENAME_COLUMN, const.CSV_CLOT_COLUMN])


# ToDo: increase val size
def create_data_split(path, train_size=0.8, valid_size=0.1, test_size=0.1):
    """Loads a CSV as df, splits the two-column df into a train, validation, test set while balancing the y. Combines
     the x and y series back into dfs and exports to CSV."""

    if train_size + valid_size + test_size != 1:
        raise ValueError("The sum of the data split should add to 1.")
    train_size = train_size/(1-test_size)

    df = load_image_filenames(path)
    X_all = df[const.CSV_FILENAME_COLUMN]
    y_all = df[const.CSV_CLOT_COLUMN]

    x_train_val, x_test, y_train_val, y_test = train_test_split(X_all, y_all, test_size=test_size,
                                                                stratify=y_all, random_state=123)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, train_size=train_size,
                                                      stratify=y_train_val, random_state=123)

    train_set = x_train.to_frame().join(y_train)
    val_set = x_val.to_frame().join(y_val)
    test_set = x_test.to_frame().join(y_test)

    print(f"full: {len(X_all)}; train + val: {len(x_train_val)}; train: {len(x_train)}; "
          f"val: {len(x_val)}' test: {len(x_test)})")

    parent_folder = os.path.dirname(path)
    train_set.to_csv(os.path.join(parent_folder, 'train_set.csv'), index=False)
    val_set.to_csv(os.path.join(parent_folder, 'val_set.csv'), index=False)
    test_set.to_csv(os.path.join(parent_folder, 'test_set.csv'), index=False)
    print("Completed data split.")


def normalise_image(image):
    image_min = image.min()
    image_max = image.max()
    image.clamp_(min=image_min, max=image_max)
    image.add_(-image_min).div_(image_max - image_min + 1e-5)
    return image


def show_image(image, label, normalise=True):
    if normalise:
        image = normalise_image(image)
    plt.imshow(image.permute(1, 2, 0), cmap='gray', vmin=0, vmax=255)
    plt.title(label)
    plt.show()


def setup_gpu():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    return device

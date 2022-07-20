from utils import config as const

import os
import torch
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split


def get_image_filenames(image_directory, csv_output):
    image_files = os.listdir(image_directory)
    df = pd.DataFrame(image_files, columns=[const.CSV_FILENAME_COLUMN])
    df.to_csv(csv_output, index=False)


def load_image_filenames(path):
    return pd.read_csv(path, usecols=[const.CSV_FILENAME_COLUMN, const.CSV_CLOT_COLUMN])


def create_data_split(path):
    df = load_image_filenames(path)
    X_all = df[const.CSV_FILENAME_COLUMN]
    y_all = df[const.CSV_CLOT_COLUMN]

    x_train_val, x_test, y_train_val, y_test = train_test_split(X_all, y_all, train_size=0.9, stratify=y_all)
    x_train, x_val, y_train, y_val = train_test_split(x_train_val, y_train_val, test_size=0.09, stratify=y_train_val)

    print(f"full: {len(X_all)}; train + val: {len(x_train_val)}; train: {len(x_train)}; "
          f"val: {len(x_val)}' test: {len(x_test)})")
    print()



def show_image(image, label):
    plt.imshow(image.permute(1, 2, 0), cmap='gray', vmin=0, vmax=255)
    plt.title(label)
    plt.show()


def setup_gpu():
    torch.cuda.empty_cache()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("Using {} device.".format(device))
    return device

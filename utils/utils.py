from utils import config

import os
import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report


def get_image_filenames(image_directory, csv_output):
    image_files = os.listdir(image_directory)
    df = pd.DataFrame(image_files, columns=[config.CSV_FILENAME_COLUMN])
    df[config.CSV_CLOT_COLUMN] = ""
    df[config.CSV_DAY_COLUMN] = ""
    df.to_csv(csv_output, index=False)
    print("Image file names written to file")


def change_image_file_resolutions(path, to_replace, new_value):
    df = pd.read_csv(path, usecols=[config.CSV_FILENAME_COLUMN, config.CSV_CLOT_COLUMN, config.CSV_DAY_COLUMN])
    df[config.CSV_FILENAME_COLUMN] = df[config.CSV_FILENAME_COLUMN].str.replace(to_replace, new_value)
    df.to_csv(path, index=False)
    print(f"Image file names written to file with '{to_replace}' changed to '{new_value}' in the filenames.")


def load_image_filenames(path, col='all'):
    if col == config.CSV_CLOT_COLUMN:
        return pd.read_csv(path, usecols=[config.CSV_FILENAME_COLUMN, config.CSV_CLOT_COLUMN])
    elif col == config.CSV_DAY_COLUMN:
        return pd.read_csv(path, usecols=[config.CSV_FILENAME_COLUMN, config.CSV_DAY_COLUMN])
    return pd.read_csv(path, usecols=[config.CSV_FILENAME_COLUMN, config.CSV_CLOT_COLUMN, config.CSV_DAY_COLUMN])


def create_data_split(path, col, train_size=0.8, valid_size=0.1, test_size=0.1):
    """Loads a CSV as df, splits the two-column df into a train, validation, test set while balancing the y. Combines
     the x and y series back into dfs and exports to CSV."""

    train_size = train_size/(1-test_size)

    df = load_image_filenames(path, col)
    X_all = df[config.CSV_FILENAME_COLUMN]
    y_all = df[col]

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


def calculate_binary_metrics(labels, pred_labels, as_dict=False):
    accuracy = torch.sum(pred_labels == labels)/len(pred_labels)
    tn, fp, fn, tp = confusion_matrix(labels, pred_labels).ravel()
    sensitivity = tp / (tp + fn)  # aka recall
    specificity = tn / (fp + tn)
    g_mean = np.sqrt(sensitivity*specificity)  # Fowlkes–Mallows index
    precision = tp / (tp + fp)
    f_score = (2 * precision * sensitivity) / (precision + sensitivity)
    jaccard_index = tp / (tp+fp+fn)
    if as_dict:
        return {'acc': accuracy, 'gmean':g_mean, 'fscore': f_score, 'jindex':jaccard_index}
    return accuracy, g_mean, f_score, jaccard_index


def calculate_multi_metrics(labels, pred_labels, as_dict=False):
    report = classification_report(labels, pred_labels, output_dict=True)
    accuracy = report['accuracy']
    f_score = report['weighted avg']['f1-score']

    if as_dict:
        return {'acc': accuracy, 'fscore': f_score}
    return accuracy, f_score

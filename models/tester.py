from utils.utils import calculate_binary_metrics, calculate_multi_metrics,normalise_image

import math
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import torch.nn.functional as F
import numpy as np


def get_predictions(model, dataloader, device):
    model.eval()

    images = []
    labels = []
    probs = []

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y_pred, _ = model(x)
            y_prob = F.softmax(y_pred, dim=-1)

            images.append(x.cpu())
            labels.append(y.cpu())
            probs.append(y_prob.cpu())

    images = torch.cat(images, dim=0)
    labels = torch.cat(labels, dim=0)
    probs = torch.cat(probs, dim=0)
    pred_labels = torch.argmax(probs, 1)

    return images, labels, probs, pred_labels


def plot_confusion_matrix(labels, pred_labels, classes):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(1, 1, 1)
    cm = confusion_matrix(labels, pred_labels)
    cm = ConfusionMatrixDisplay(cm, display_labels=classes)
    cm.plot(values_format='d', cmap='Blues', ax=ax)
    if len(classes) == 2:
        acc, gmean, fscore, j_index = calculate_binary_metrics(labels, pred_labels)
        plt.title(f'Accuracy: {acc:.2f}, G-mean: {gmean:.2f}, F Score: {fscore:.2f}')
    else:
        acc, fscore = calculate_multi_metrics(labels, pred_labels)
        plt.title(f'Accuracy: {acc:.2f}, F Score: {fscore:.2f}')
    plt.xticks(rotation=20)
    plt.show()


def get_incorrect_examples(images, labels, probs, pred_labels):
    corrects = torch.eq(labels, pred_labels)

    incorrect_examples = []

    for image, label, prob, pred_label, correct in zip(images, labels, probs, pred_labels, corrects):
        if not correct:
            incorrect_examples.append((image, label, prob, pred_label))
    incorrect_examples.sort(reverse=True, key=lambda x: torch.max(x[2], dim=0).values)
    return incorrect_examples


def plot_most_incorrect(images, labels, probs, pred_labels, classes, n_images=10, normalise=True):

    incorrect = get_incorrect_examples(images, labels, probs, pred_labels)

    if n_images > len(incorrect):
        n_images = len(incorrect)

    rows = math.floor(np.sqrt(n_images))
    cols = math.ceil(np.sqrt(n_images))

    if rows * cols < n_images:
        cols += 1

    fig = plt.figure(figsize=(25, 20))

    for i in range(rows * cols):

        ax = fig.add_subplot(rows, cols, i + 1)

        image, true_label, prob, pred_label = incorrect[i]
        image = image.permute(1, 2, 0)
        true_prob = prob[true_label]
        incorrect_prob = prob[pred_label]
        true_class = classes[true_label]
        incorrect_class = classes[pred_label]

        if normalise:
            image = normalise_image(image)

        ax.imshow(image.cpu().numpy())
        ax.set_title(f'true label: {true_class} ({true_prob:.3f})\n '
                     f'pred label: {incorrect_class} ({incorrect_prob:.3f})',
                     fontdict={'fontsize': 40})
        ax.axis('off')

    fig.subplots_adjust(hspace=0.4)
    plt.show()

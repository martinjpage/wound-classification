from torchvision.models import ResNet50_Weights, ResNet101_Weights
from torchvision import transforms
import torch
import random


def train_transformer(model):
    torch.random.manual_seed(123)
    random.seed(123)
    composed = [transforms.RandomRotation(degrees=5), transforms.RandomHorizontalFlip(p=0.5),
                transforms.RandomVerticalFlip(p=0.5), transforms.GaussianBlur(kernel_size=1),
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0)]

    if model == 'resnet50':
        composed.extend([ResNet50_Weights.IMAGENET1K_V2.transforms()])
    elif model == 'resnet101':
        composed.extend([ResNet101_Weights.IMAGENET1K_V2.transforms()])
    return transforms.Compose(composed)


def val_transformer(model):
    torch.random.manual_seed(123)
    random.seed(123)

    composed = [transforms.RandomRotation(degrees=5), transforms.RandomHorizontalFlip(p=0.5),
                transforms.GaussianBlur(kernel_size=1)]

    if model == 'resnet50':
        composed.extend([ResNet50_Weights.IMAGENET1K_V2.transforms()])
    elif model == 'resnet101':
        composed.extend([ResNet101_Weights.IMAGENET1K_V2.transforms()])
    return transforms.Compose(composed)


def test_transformer(model):
    torch.random.manual_seed(123)
    random.seed(123)

    composed = []

    if model == 'resnet50':
        composed.extend([ResNet50_Weights.IMAGENET1K_V2.transforms()])
    elif model == 'resnet101':
        composed.extend([ResNet101_Weights.IMAGENET1K_V2.transforms()])
    return transforms.Compose(composed)

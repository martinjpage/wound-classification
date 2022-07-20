from utils import config

from torchvision import transforms


def train_transformer():
    return transforms.Compose([transforms.Resize([config.SIZE, config.SIZE]),
                               transforms.ToTensor(),
                               transforms.Normalize(config.MEAN, config.STD)])


def val_transformer():
    return transforms.Compose([transforms.Resize([config.SIZE, config.SIZE]),
                               transforms.ToTensor(),
                               transforms.Normalize(config.MEAN, config.STD)])


def test_transformer():
    return None

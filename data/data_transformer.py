from utils import config

from torchvision import transforms


def transform_data():
    return transforms.Compose([transforms.Resize([config.SIZE, config.SIZE]),
                               transforms.ToTensor(),
                               transforms.Normalize(config.MEAN, config.STD)])

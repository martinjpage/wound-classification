from utils import config

from torchvision import transforms

# ToDo: ResNet50_Weights.IMAGENET1K_V2.transforms - but is order important?
# ToDo: scaling pixel values: transforms.Lambda(lambda t: t/255)  # Scale pixel values to 0..1

def train_transformer():
    return transforms.Compose([
        transforms.Resize(size=config.PRETRAINED_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomRotation(degrees=5),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomCrop(size=config.PRETRAINED_CROP, padding=10),
        transforms.GaussianBlur(kernel_size=1),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0, hue=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=config.PRETRAINED_MEANS, std=config.PRETRAINED_STDS)
    ])


def val_transformer():
    return transforms.Compose([
        transforms.Resize(config.PRETRAINED_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.RandomRotation(5),
        transforms.RandomHorizontalFlip(0.5),
        transforms.RandomCrop(config.PRETRAINED_CROP, padding=10),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t / 255),
        transforms.Normalize(mean=config.PRETRAINED_MEANS,
                             std=config.PRETRAINED_STDS)
    ])


def test_transformer():
    return transforms.Compose([
        transforms.Resize(config.PRETRAINED_SIZE, interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.CenterCrop(config.PRETRAINED_CROP),
        transforms.ToTensor(),
        transforms.Lambda(lambda t: t / 255),
        transforms.Normalize(mean=config.PRETRAINED_MEANS,
                             std=config.PRETRAINED_STDS)
    ])

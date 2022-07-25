from utils import config
from models.resnet import create_resnet_config, ResNet, Bottleneck

import torch.nn as nn
from torchvision.models import resnet50, ResNet50_Weights


# ToDo: compare to 101
def create_resnet50():
    ResNetConfig = create_resnet_config()
    resnet50_config = ResNetConfig(block=Bottleneck,
                                   n_blocks=[3, 4, 6, 3],
                                   channels=[64, 128, 256, 512])

    pretrained_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
    IN_FEATURES = pretrained_model.fc.in_features
    fc = nn.Linear(IN_FEATURES, config.OUTPUT_DIM)
    pretrained_model.fc = fc
    model = ResNet(resnet50_config, config.OUTPUT_DIM)
    model.load_state_dict(pretrained_model.state_dict())
    return model

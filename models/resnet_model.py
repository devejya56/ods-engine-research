import torch.nn as nn
import torchvision.models as tv_models

def get_resnet18(num_classes=10, input_channels=3, small_input=True):
    """
    ResNet-18 adapted for various resolutions.
    - small_input=True: Adapted for 32x32 (CIFAR) - 3x3 conv1, no initial maxpool.
    - small_input=False: Standard ImageNet-style - 7x7 conv1, maxpool enabled.
    """
    model = tv_models.resnet18(weights=None)
    
    if small_input:
        # Adapt for small-resolution inputs (CIFAR-10 is 32x32)
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
    else:
        # Standard configuration, just update input channels
        model.conv1 = nn.Conv2d(input_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)

    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    return model

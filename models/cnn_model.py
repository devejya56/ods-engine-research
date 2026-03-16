import torch.nn as nn
import torch.nn.functional as F

class BaselineCNN(nn.Module):
    """
    Baseline CNN as specified for Experiment 1.
    Now supports various input resolutions and channels.
    """
    def __init__(self, num_classes=10, input_channels=3, input_size=32):
        super(BaselineCNN, self).__init__()
        
        # 1. Conv Layer 1: Filters 32, Kernel 3x3, ReLU
        self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
        # 2. Max Pooling: Kernel 2x2
        self.pool1 = nn.MaxPool2d(2, 2)
        
        # 3. Conv Layer 2: Filters 64, Kernel 3x3, ReLU
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        # 4. Max Pooling: Kernel 2x2
        self.pool2 = nn.MaxPool2d(2, 2)
        
        # 5. Conv Layer 3: Filters 128, Kernel 3x3, ReLU
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        
        # 6. Flatten logic
        # After two 2x2 poolings, input_size becomes input_size // 4.
        self.feature_map_size = input_size // 4
        self.fc1 = nn.Linear(128 * self.feature_map_size * self.feature_map_size, 128)
        
        # 8. Output Layer: Units 10
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool1(F.relu(self.conv1(x)))
        x = self.pool2(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        # 7. FC Layer: Units 128, ReLU
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

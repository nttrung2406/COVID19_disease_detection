import os, sys
sys.path.append('C:/Users/flori/OneDrive/Máy tính/Tai-lieu/HCMUS/Image processing')
from lib import *

class CNN(Module):
    def __init__(self, num_classes=3) -> None:
        super().__init__()

        # First Convolutional Layer
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.3, True)
        )

        # Second Convolutional Layer
        self.conv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)),
            nn.Dropout2d(0.5, True)
        )

        # Third Convolutional Layer
        self.conv3 = nn.Sequential(
            nn.Conv2d(64, 32, kernel_size=3, padding=1),
            nn.ReLU(True),
            nn.MaxPool2d((2, 2)), 
        )

        # Fourth Convolutional Layer
        self.conv4 = nn.Sequential(
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        # Fifth Convolutional Layer
        self.conv5 = nn.Sequential(
            nn.Conv2d(32, 16, kernel_size=3, padding=1),
            nn.ReLU(True)
        )

        # Combine the layers into a model
        self.model = nn.Sequential(
            self.conv1,
            self.conv2,
            self.conv3,
            self.conv4,
            self.conv5,
            nn.AdaptiveAvgPool2d(1),  # Global Average Pooling to make output size 16x1x1
            nn.Flatten(),  
            nn.Linear(16, 512), 
            nn.Dropout(0.2, True),
            nn.Linear(512, num_classes)
        )

    def forward(self, img: Tensor) -> Tensor:
        return self.model(img)



import numpy as np
from util.vearchutil import VearchUtil
from util.testutil import TestUtil
import cv2
import matplotlib.pyplot as plt
from PIL import Image
import io
from pprint import pprint
import torchvision.transforms.functional as TF
import random
import torch
import torch.nn as nn
from torchvision.models.alexnet import AlexNet
from torchvision import datasets, transforms, models

class AlphaBgResize:
    """Adjsut image size based on alpha mask"""

    def __init__(self):
        pass

    def __call__(self, x):
        if x.shape[2] == 3:
            # only 3 channel return 
            return x

        arr = x[:,:,3]
        idx = np.transpose(np.nonzero(arr))
        h = x.shape[0]
        w = x.shape[1]
        lt = min(idx[0])
        bt = max(idx[-1])

        lt = lt - 10 if lt-10 > 0 else 0
        bt = bt + 10 if bt+ 10 < max(h,w) else max(h,w)

        croped_image = x[lt:bt,lt:bt,:]

        transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            # transforms.Normalize(mean=[0.485], std=[0.229]),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        x = transform(croped_image)

        return croped_image

class ParameterError(Exception):
    pass

class AlphaAlexNet(AlexNet):
    """Adding Alpha BG info as a descriptor of shapes

    Args:
        AlexNet (num_classes): default 1000
                droput: default 0.5
    """
    def __init__(self, num_classes: int = 1000,dropout: float = 0.5) -> None:        
        super().__init__(num_classes=num_classes,dropout=dropout)
    
        self.features = nn.Sequential(
            nn.Conv2d(4, 64, kernel_size=11, stride=4, padding=2), # channel 3->4
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[2] <=3:
            raise ParameterError("AlphaAlexNet expecting input is 224*224*4 RGBA image as ndarray")
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

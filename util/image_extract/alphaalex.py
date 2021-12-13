# Copyright 2019 The Vearch Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# ==============================================================================

import cv2
import numpy as np
from PIL import Image
import torch
from torchvision import transforms
import torchvision.models as models
import torch.nn.functional as F
from util.trainingutil import AlphaAlexNet, AlphaBgTransform


class BaseModel(object):
    def __init__(self):
        self.image_size = 224
        self.dimision = 512
        self.load_model()

    def load_model(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = AlphaAlexNet()
        trained_model = "./models/alexnet_alpha.pth"
        self.model.load_state_dict(torch.load(trained_model))
        self.model = self.model.eval()
        self.model.to(device=self.device)

    def preprocess_input(self, image):
        # input PIL image RGB 3 channel
        # transform = transforms.Compose(
        #     [
        #         transforms.Resize((224, 224)),
        #         transforms.ToTensor(),
        #         transforms.Normalize(
        #             mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        #         ),
        #     ]
        # )

        transform = transforms.Compose([            
            AlphaBgTransform(alpha=False),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        x = transform(image).to(self.device)

        return x

    def forward(self, x):
        x = self.preprocess_input(x).unsqueeze(0)
        x = self.model.features(x)
        x = F.max_pool2d(x, kernel_size=(6, 6))
        x = x.view(x.size(0),-1)
        return self.torch2list(x)

    def torch2list(self, torch_data):
        return torch_data.cpu().detach().numpy().tolist()


def load_model():
    return BaseModel()


if __name__ == "__main__":
    print("__main__")

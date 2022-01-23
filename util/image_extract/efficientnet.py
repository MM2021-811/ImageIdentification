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
from efficientnet_pytorch import EfficientNet
from torch.nn import Linear, Sequential
import torch.nn.functional as F
from collections import OrderedDict
import os

# os.environ['CUDA_VISIBLE_DEVICES'] ='3'


class BaseModel(object):

    def __init__(self):
        self.image_size = 224
        self.dimision = 1280
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = torch.device("cpu")
        # self.fc = Linear(1280, 512).to(self.device)
        self.load_model()
   
    def load_model(self):
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = EfficientNet.from_pretrained('efficientnet-b0').to(self.device)

        # self.model = EfficientNet.from_pretrained('efficientnet-b0', include_top=False).to(self.device)
        self.model = self.model.eval()
        self.PIXEL_MEANS = torch.tensor((0.485, 0.456, 0.406)).to(self.device)
        self.PIXEL_STDS = torch.tensor((0.229, 0.224, 0.225)).to(self.device)
        self.num = torch.tensor(255.0).to(self.device)

    def preprocess_input(self, image):
        image = cv2.resize(image, (self.image_size, self.image_size))
        # gpu version
        image_tensor = torch.from_numpy(image.copy()).to(self.device).float()
        image_tensor /= self.num
        image_tensor -= self.PIXEL_MEANS
        image_tensor /= self.PIXEL_STDS
        image_tensor = image_tensor.permute(2, 0, 1)
        return image_tensor

    # define a function to reduce the dimension to 512
    # def block_dim_red(in_features):
    #     block_dim_red = Sequential(OrderedDict([
    #         ('fc', Linear(1280, 512)),
    #     ]))
    #     return block_dim_red

    # def fc():
    #     fc = Linear(in_features = 1280, out_feaures = 512)
    #     return fc

    def forward(self, x):
        # x = torch.stack(x)
        # x = x.to(self.device)
        x = self.preprocess_input(x).unsqueeze(0)
        # modified for efficient net
        # extraccted feature shape torch.Size([1, 1280, 7, 7]), Efficientnet git.readme
        x = self.model.extract_features(x)
        x = F.max_pool2d(x, kernel_size=(7, 7))
        x = x.view(x.size(0),-1)
        # x = torch.reshape(x,(-1,1))
        # x = self.fc(x) # fully connecte layer to reduce dimension
        # x = torch.reshape(x, (-1,1)) # sun added
        # print(x.shape)
        # x = torch.squeeze(x,-1)
        # x = torch.squeeze(x,-1)
        return self.torch2list(x)
        # return x

    def torch2list(self, torch_data):
        return torch_data.cpu().detach().numpy().tolist()

def load_model():
    return BaseModel()

def main():
    model = load_model()
    model.load_model()
    import urllib.request
    
    url = "https://www.planetware.com/wpimages/2020/02/france-in-pictures-beautiful-places-to-photograph-eiffel-tower.jpg"
    
    resp = urllib.request.urlopen(url).read()
    image = np.asarray(bytearray(resp), dtype="uint8")
    image = cv2.imdecode(image, cv2.IMREAD_COLOR)
    feat = model.forward(image)
    feat = np.array(feat)
    # feat_test = feat[0]/np.linalg.norm(feat[0])

    # print(feat_test)
    # print(feat_test.shape)
    print(feat)
    print(feat.shape)

if __name__ == "__main__":
    main()

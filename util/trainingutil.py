#
# AlphaAlexNet code is based on torchvision AlexNet
#
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
import rembg.bg as rembg
from util.exposure_enhancement import enhance_image_exposure

class AlphaBgTransform:
    """Adjsut image size based on alpha mask
    Output 224*224*4 ndarray
    """

    def __init__(self):
        pass

    def __call__(self, x):
        if x.shape[2] == 3:
            # only 3 channel remove bg and add alpha channel
            x = self.remove_bg(x)

        #crop
        x= self.center_crop(x)

        # tosquare
        x = self.to_square(x)

        # resize
        x = cv2.resize(x, (224,224), interpolation = cv2.INTER_AREA)

        #enhance color
        x1 = self.enhance_color(x[:,:,:-1])
        x[:,:,:-1] = x1

        # basic transform for the model
        transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406,0.9], std=[0.229, 0.224, 0.225,0.2]),
        ])
        x = transform(x)

        #transform changed dimention to 2,0,1 which is 4 * 244 * 244

        return x

    def remove_bg(self,
        data,
        model_name="u2net",
        alpha_matting=False,
        alpha_matting_foreground_threshold=240,
        alpha_matting_background_threshold=10,
        alpha_matting_erode_structure_size=10,
        alpha_matting_base_size=1000,
        ):
        """The remove bg code is based on: https://github.com/ziyunxiao/rembg
           For Original code and usage please check their Github Repo
        Args:
            data ([type]): [description]
            model_name (str, optional): [description]. Defaults to "u2net".
            alpha_matting (bool, optional): [description]. Defaults to False.
            alpha_matting_foreground_threshold (int, optional): [description]. Defaults to 240.
            alpha_matting_background_threshold (int, optional): [description]. Defaults to 10.
            alpha_matting_erode_structure_size (int, optional): [description]. Defaults to 10.
            alpha_matting_base_size (int, optional): [description]. Defaults to 1000.
        Returns:
            [type]: [description]
        """
        model = rembg.get_model(model_name)
        # img = Image.open(io.BytesIO(data)).convert("RGB")
        img = data # input must be RGB Ndarray
        mask = rembg.detect.predict(model, np.array(img)).convert("L")

        if alpha_matting:
            try:
                cutout = rembg.alpha_matting_cutout(
                    img,
                    mask,
                    alpha_matting_foreground_threshold,
                    alpha_matting_background_threshold,
                    alpha_matting_erode_structure_size,
                    alpha_matting_base_size,
                )
            except:
                cutout = rembg.naive_cutout(img, mask)
        else:
            cutout = rembg.naive_cutout(img, mask)

        return cutout
    
    def center_crop(self,x):
        arr = x[:,:,3]
        idx = np.transpose(np.nonzero(arr))
        h = x.shape[0]
        w = x.shape[1]
        lt = min(idx[0])
        bt = max(idx[-1])

        lt = lt - 10 if lt-10 > 0 else 0
        bt = bt + 10 if bt+ 10 < max(h,w) else max(h,w)

        croped_image = x[lt:bt,lt:bt,:]

        return croped_image

    def enhance_color(self,x):
        gamma = 0.6
        lambda_ = 0.15
        sigma=3
        bc =1
        bs=1
        be = 1
        eps = 1e-3

        # correct color
        enhanced_image = enhance_image_exposure(x, gamma, lambda_, dual=True,
                                        sigma=sigma, bc=bc, bs=bs, be=be, eps=eps)
        return enhanced_image

    def to_square(self,x):
        (h, w,c) = x.shape

        # padding
        dim = max(h,w)
        img = np.zeros((dim,dim,c))

        if h>w:
            s_idx = (h-w)//2
            e_idx = s_idx + w
            img[:,s_idx:e_idx,:] = x
        elif h< w:
            s_idx = (w-h)//2
            e_idx = s_idx + h
            img[s_idx:e_idx,:,:] = x
        else:
            img = x
        return img

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
        if x.shape[0] <=3:
            raise ParameterError("AlphaAlexNet expecting input is 4*224*224 ARGB tensor")
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

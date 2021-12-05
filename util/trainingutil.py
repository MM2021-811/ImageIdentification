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
import torch.nn.functional as F

from torchvision.models.alexnet import AlexNet
from torchvision import datasets, transforms, models
import rembg.bg as rembg
from util.exposure_enhancement import enhance_image_exposure, get_under_n_over_channel

class AlphaBgTransform:
    """Adjsut image size based on alpha mask
    Output 224*224*4 ndarray
    """

    def __init__(self, alpha=True):
        self.u2net = rembg.get_model("u2net")
        self.alpha = alpha

    def __call__(self, x):
        if type(x) is Image.Image:
            # conver to opencv image, by default it is RGB or RGBA
            x = np.array(x)
        
        if x.shape[2] == 3:
            # only 3 channel remove bg and add alpha channel
            x = self.remove_bg(x)

        #crop
        x = AlphaBgTransform.center_crop(x)

        # tosquare
        x = AlphaBgTransform.to_square(x)

        # resize
        # x = cv2.resize(x, (224,224), interpolation = cv2.INTER_AREA)
        x = AlphaBgTransform.resize(x,224)

        # #enhance color
        # x = AlphaBgTransform.enhance_color(x)

        # (cu,co) = get_under_n_over_channel(im=x[:,:,:-1])

        # basic transform for the model
        transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406,0.9], std=[0.229, 0.224, 0.225,0.2]),
        ])
        x = transform(x)

        #transform changed dimention to 2,0,1 which is 4 * 244 * 244
        if self.alpha is False:
            # return 3 channel ndarray
            x = x[:-1,:,:]

        return x

    @staticmethod
    def resize(x, dim):
        x = cv2.resize(x, (dim,dim), interpolation = cv2.INTER_AREA)
        return x

    def remove_bg(self,
        data,
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
        # img = Image.open(io.BytesIO(data)).convert("RGB")
        img = Image.fromarray(data) # input must RGB NDarray
        mask = rembg.detect.predict(self.u2net, np.array(img)).convert("L")
        # mask = rembg.detect.predict(self.u2net, img).convert("L")

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

        return np.array(cutout)
    
    @staticmethod
    def center_crop(x):
        arr = x[:,:,3]
        idx = np.transpose(np.nonzero(arr))
        h = x.shape[0]
        w = x.shape[1]
        
        lh = min(idx[:,0]) - 10
        lw = min(idx[:,1]) - 10
        bh = max(idx[:,0]) + 10
        bw = max(idx[:,1]) + 10
        
        lh = 0 if lh < 0 else lh
        lw = 0 if lw < 0 else lw
        bh = h if bh > h else bh
        bw = w if bw > w else bw

        croped_image = x[lh:bh,lw:bw,:]

        return croped_image

    @staticmethod
    def enhance_color(x):
        x1 = x[:,:,:-1]

        gamma = 0.6
        lambda_ = 0.15
        sigma=3
        bc =1
        bs=1
        be = 1
        eps = 1e-3

        # correct color
        enhanced_image = enhance_image_exposure(x1, gamma, lambda_, dual=True,
                                        sigma=sigma, bc=bc, bs=bs, be=be, eps=eps)

        x[:,:,:-1] = enhanced_image

        return x

    @staticmethod
    def to_square(x):
        (h, w,c) = x.shape

        # padding
        dim = max(h,w)
        img = np.zeros((dim,dim,c),dtype="uint8")

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

class AlphaAlexNet(nn.Module):
    """Adding Alpha BG info as a descriptor of shapes

    Args:
        AlexNet (num_classes): default 1000
                droput: default 0.5
    """
    def __init__(self, num_classes: int = 1000,dropout: float = 0.5) -> None:        
        super(AlphaAlexNet,self).__init__()
    
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
        if x.shape[1] <=3:
            raise ParameterError("AlphaAlexNet expecting input is 4*224*224 ARGB tensor")
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlphaWeightedAlexNet(nn.Module):
    def __init__(self,dropout: float = 0.5) -> None:        
        super().__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.eval()
    
        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.feat_layer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 2048),
            nn.ReLU(inplace=True),
            nn.Linear(2048, 1024),
        )

        self.discritor = nn.Sequential(
            nn.Linear(2 * 1024, 1024),
            nn.Linear(1024, 512),
            nn.Linear(512, 2),
            nn.Softmax(1)
        )

    def _features(self,x):
        with torch.no_grad():
            x = self.alexnet.features(x)
        return x

    def features(self,x):
        x = self._features(x)
        x = self.avgpool(x)
        x = x.view(x.size()[0],-1)
        x = self.feat_layer(x)
        return x

    def forward(self, input1, input2):
        output1 = self.features(input1)
        output1 = output1.view(output1.size()[0], -1)#make it suitable for fc layer.
        output2 = self.features(input2)
        output2 = output2.view(output2.size()[0], -1)
        
        output = torch.cat((output1, output2),1)
        output = self.discritor(output)
        return output
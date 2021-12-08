#
# AlphaAlexNet code is based on torchvision AlexNet
#
import numpy as np
from pymatting.util.util import normalize
from torch._C import device
from util.vearchutil import VearchUtil
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
import json
import pickle
import os
import PIL

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
        x = AlphaBgTransform.enhance_color(x)

        # (cu,co) = get_under_n_over_channel(im=x[:,:,:-1])

        # basic transform for the model
        # transform = transforms.Compose([
        #     # transforms.Resize(224),
        #     # transforms.CenterCrop(224),
        #     transforms.ToTensor(),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406,0.9], std=[0.229, 0.224, 0.225,0.2]),
        # ])
        # x = transform(x)

        #transform changed dimention to 2,0,1 which is 4 * 244 * 244
        if self.alpha is False:
            # return 3 channel ndarray
            # x = x[:-1,:,:]
            x = x[:,:,:-1]

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

    @staticmethod
    def de_normalize(t:torch.tensor,mean,std):
        inv_normalize = transforms.Normalize(
            mean= [-m/s for m, s in zip(mean, std)],
            std= [1/s for s in std]
            )
        return inv_normalize

class ParameterError(Exception):
    pass

class AlexNet(nn.Module):
    def __init__(self,device=None):
        super(AlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels= 96, kernel_size= 11, stride=4, padding=2 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 9216, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=10)

    def forward(self,x):

        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlphaAlexNet(nn.Module):
    """Adding Alpha CU and CO channel info as a descriptor of shapes

    Args:
        AlphaAlexNet 
    """
    def __init__(self):
        super(AlphaAlexNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=9, out_channels= 96, kernel_size= 11, stride=4, padding=2 )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2)
        self.conv2 = nn.Conv2d(in_channels=96, out_channels=256, kernel_size=5, stride= 1, padding= 2)
        self.conv3 = nn.Conv2d(in_channels=256, out_channels=384, kernel_size=3, stride= 1, padding= 1)
        self.conv4 = nn.Conv2d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1)
        self.conv5 = nn.Conv2d(in_channels=384, out_channels=256, kernel_size=3, stride=1, padding=1)
        self.fc1  = nn.Linear(in_features= 9216, out_features= 4096)
        self.fc2  = nn.Linear(in_features= 4096, out_features= 4096)
        self.fc3 = nn.Linear(in_features=4096 , out_features=10)

    def _getcuco(self,x):
        #convert from 3 channels to 9 channels
        mean=[0.485, 0.456, 0.406]
        std=[0.229, 0.224, 0.225]
        inv_normalize = transforms.Normalize(
            mean= [-m/s for m, s in zip(mean, std)],
            std= [1/s for s in std]
            )
        norm = transforms.Normalize(mean,std)

        device = x.device
        (d,c,h,w) = x.shape
        result = torch.tensor(np.zeros((d,9,h,w),dtype="float32")).to(device)
        for i in range(d):
            x1 = x[i,:,:,:]
            y = inv_normalize(x1)
            y = y.permute(1,2,0) * 255
            y = y.cpu().detach().numpy()
            # (cu,co) = get_under_n_over_channel(im=y)            
            # cu = torch.tensor(np.array(cu),dtype=torch.float32).permute(2,0,1)
            # co = torch.tensor(np.array(co),dtype=torch.float32).permute(2,0,1)
            # x2 = norm(cu).to(device)    
            # x3 = norm(co).to(device)

            # testing model
            x2 = torch.transpose(x1,dim0=1,dim1=2)
            y = Image.fromarray(np.uint8(y))
            x3 = PIL.ImageOps.invert(y)
            x3 = torch.tensor(np.array(x3),dtype=torch.float32).permute(2,0,1)
            x3 = norm(x3)

            result[i,0:3,:,:] = x1
            result[i,3:6,:,:] = x2
            result[i,6:9,:,:] = x3
        
        return result

    def features(self,x):
        # convert 3 channel to 9 chanel
        x = self._getcuco(x)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        
        return x
    def forward(self,x):
        # convert 3 channel to 9 chanel
        x = self._getcuco(x)
        x = F.relu(self.conv1(x))
        x = self.maxpool(x)
        x = F.relu(self.conv2(x))
        x = self.maxpool(x)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = self.maxpool(x)
        x = x.reshape(x.shape[0], -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class AlphaWeightedVgg16Net(nn.Module):
    def __init__(self,device="cpu",dropout: float = 0.5) -> None:
        super().__init__()
        self.vgg16 = models.vgg16(pretrained=True)
        self.vgg16.to(device)
        self.vgg16.eval()

        self.pool = nn.MaxPool2d(kernel_size=(7, 7))
        self.feat_layer = nn.Sequential(
            nn.Linear(512, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            # nn.Linear(4096, 2048),
            # nn.ReLU(inplace=True),
            #  nn.Dropout(p=dropout),
            # nn.Linear(2048, 1024),
            # nn.ReLU(inplace=True),
            nn.Linear(4096, 512),
        )
        
        self.discritor = nn.Sequential(
            nn.Linear(2 * 512, 512),
            nn.Linear(512, 2),
            nn.Softmax(1)
        )

    def _features(self,x):
        with torch.no_grad():
            x = self.vgg16.features(x)
        return x

    def features(self,x):
        x = self._features(x)
        x = self.pool(x)
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


class AlphaWeightedAlexNet(nn.Module):
    def __init__(self,device="cpu",dropout: float = 0.5) -> None:
        super().__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.to(device)
        self.alexnet.eval()

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.feat_layer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 512),
        )

        self.discritor = nn.Sequential(
            nn.Linear(2 * 512, 4096),
            nn.Linear(4096, 2),
            # nn.Softmax(1)
            nn.Sigmoid()
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

    def forward(self, x1):

        x1 = self.features(x1)
        x1 = x1.view(x1.size()[0], -1)#make it suitable for fc layer.

        x2 = self.features(x2)
        x2 = x2.view(x2.size()[0], -1)#make it suitable for fc layer.

        x3 = self.features(x3)
        x3 = x3.view(x3.size()[0], -1)#make it suitable for fc layer.

        output = torch.cat((x1, x2),1)
        output = self.discritor(output)
        return output

class SiameseAlexNet(nn.Module):
    def __init__(self,device="cpu",dropout: float = 0.5) -> None:
        super().__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.to(device)
        self.alexnet.eval()

        self.avgpool = nn.AdaptiveAvgPool2d((6, 6))
        self.feat_layer = nn.Sequential(
            nn.Linear(256 * 6 * 6, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(4096, 4096),
            nn.Linear(4096, 512),
        )

        self.discritor = nn.Sequential(
            nn.Linear(2 * 512, 4096),
            nn.Linear(4096, 2),
            # nn.Softmax(1)
            nn.Sigmoid()
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


class SiameseLoader(object):
    def __init__(self,data_path="./data/zerobox_nobg", train=True, batch_size=64,shuffle=True, use_cache=True) -> None:
        super().__init__()
        self.batch_size = batch_size
        self.data_path = data_path
        self.train = train
        self.use_cache = use_cache

        if self.train is True:
            meta_file = f"{self.data_path}/meta_train.json"
        else:
            meta_file = f"{self.data_path}/meta_test.json"

        c_file = f"{self.data_path}/meta_classes.json"
        self.classes = json.load(open(c_file,"r"))
        self.data = json.load(open(meta_file,"r"))

        # self._init_maping(shuffle)
        self._load_data_to_memory()
        

    def __len__(self):
        #    return len(self.data_idx)
        n = len(self.images) 
        return n * 10 * 10

    def __iter2__(self):
        batchid = 0
        i = 0
        data1 = np.zeros((self.batch_size,3,224,224),dtype="float32")
        data2 = np.zeros((self.batch_size,3,224,224),dtype="float32")
        labels = np.zeros((self.batch_size,2))
        while i < len(self.data_idx):
            j = i % self.batch_size

            arr = self.data_idx[i]
            (img1,img2,label) = self.get_batch_data(arr[0],arr[1])
            data1[j,:,:,:] = img1
            data2[j,:,:,:] = img2
            labels[j,:] = label
            i += 1
            if i % self.batch_size == 0:
                yield (data1,data2,labels)
                data1 = np.zeros((self.batch_size,3,224,224),dtype="float32")
                data2 = np.zeros((self.batch_size,3,224,224),dtype="float32")
                labels = np.zeros((self.batch_size,2))
                batchid += 1
        idx = i % self.batch_size
        if idx > 0:
            # last not whole batch
            yield (data1[:idx,:,:],data2[:idx,:,:],labels[:idx,:,:])

    def get_batch_data(self,i, j):
        img1 = self.images[i]
        img2 = self.images[j]
        l1 = self.labels[i]
        l2 = self.labels[j]
        if l1 == l2:
            label = [1,0]
        else:
            label = [0,1]
        return (img1.numpy(), img2.numpy(), label)

    def _init_maping(self,shuffle):
        n = len(self.data)
        total = (3*n)**2
        self.data_idx = []
        for i in range(3*n):
            for j in range(3*n):
                self.data_idx.append((i,j))

        if shuffle is True:
            np.random.shuffle(self.data_idx)

    def _load_data_to_memory2(self):
        cache_file = f"{self.data_path}/siamese_data_{self.train}.pkl"
        print(f"Training {self.train} cache file: {cache_file}")
        
        if self.use_cache is True:
            if os.path.exists(cache_file):
                try:
                    data = pickle.load(open(cache_file,"rb"))
                    self.images = data["images"]
                    self.labels = data["labels"]
                    return
                except:
                    pass

        al_transform = AlphaBgTransform(alpha=False)
        transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.images = []
        self.labels = []
        for item in self.data:
            label = item["class"]
            image_name = f"{self.data_path}/images/{label}/{item['file_name']}"
            img = cv2.imread(image_name,cv2.IMREAD_UNCHANGED)
            img = al_transform(img)
            (cu,co) = get_under_n_over_channel(im=img)
            img = transform(img)
            cu = transform(cu)
            co = transform(co)

            self.images.append(img)
            self.labels.append(label)
            self.images.append(cu)
            self.labels.append(label)
            self.images.append(co)
            self.labels.append(label)

        data = {"images":self.images, "labels": self.labels}
        pickle.dump(data,open(cache_file,"wb"))


    def _load_data_to_memory(self):
        cache_file = f"{self.data_path}/siamese_data_{self.train}_1.pkl"
        print(f"Training {self.train} cache file: {cache_file}")
        
        if self.use_cache is True:
            if os.path.exists(cache_file):
                try:
                    data = pickle.load(open(cache_file,"rb"))
                    self.images = data["images"]
                    self.cls_images = data["cls_images"]
                    return
                except:
                    pass

        al_transform = AlphaBgTransform(alpha=False)
        transform = transforms.Compose([
            # transforms.Resize(224),
            # transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        self.images = dict()
        self.cls_images = dict()
        for cls in self.classes:
            self.cls_images[cls] = []            

        for item in self.data:
            label = item["class"]
            key = f"{label}_{item['file_name']}"
            image_name = f"{self.data_path}/images/{label}/{item['file_name']}"
            img = cv2.imread(image_name,cv2.IMREAD_UNCHANGED)
            img = al_transform(img)
            (cu,co) = get_under_n_over_channel(im=img)
            img = transform(img)
            cu = transform(cu)
            co = transform(co)

            item["image"] = img
            item["cu"] = cu
            item["co"] = co
            self.images[key] = item
            self.cls_images[label].append(key)
        
        classes = list(self.cls_images.keys())
        for cls in classes:
            if len(self.cls_images[cls]) == 0:
                del self.cls_images[cls]

        data = {"cls_images":self.cls_images, "images":self.images}
        pickle.dump(data,open(cache_file,"wb"))

    def __iter__(self):
        n = len(self.images)        
        data1 = np.zeros((10 * n,3,224,224),dtype="float32")
        data2 = np.zeros((10 * n,3,224,224),dtype="float32")
        labels = np.zeros((10 * n ,2))
        idx = 0
        # total = self.__len__()
        cnt = 0
        for i in range(10):
            idx = 0
            for key in self.images:
                cnt += 1
                # get 5 same class images 
                # 2 from cu, co channel, 3 from other images
                item = self.images[key]
                cls = item["class"]                

                data1[idx,:,:,:] = item["image"]
                data2[idx,:,:,:] = item["cu"]
                labels[idx,:] = [1,0]
                idx += 1

                data1[idx,:,:,:] = item["image"]
                data2[idx,:,:,:] = item["co"]
                labels[idx,:] = [1,0]
                idx += 1
                
                skeys = self._get_random_same_class_image(cls,count=3)
                for skey in skeys:
                    data1[idx,:,:,:] = item["image"]
                    data2[idx,:,:,:] = self.images[skey]["image"]
                    labels[idx,:] = [1,0]
                    idx += 1

                # get 5 different class images random choose
                rkeys = self._get_random_diff_class_image(cls,count=5)
                for rkey in rkeys:
                    data1[idx,:,:,:] = item["image"]
                    data2[idx,:,:,:] = self.images[rkey]["image"]
                    labels[idx,:] = [0,1]
                    idx += 1
            
            yield (data1,data2,labels)

    def _get_random_same_class_image(self,cls,count=3):
        keys = self.cls_images[cls]
        keys = np.array(keys)
        m = np.random.choice(len(keys), size=count, replace=True)       
        return keys[m]

    def _get_random_diff_class_image(self,cls,count=5):
        classes = list(self.cls_images.keys())
        classes.sort()
        classes.remove(cls)
    
        classes = np.array(classes)
        m = np.random.choice(len(classes), size=count, replace=True)
        
        ret = []
        for cls in classes[m]:
            key = self._get_random_same_class_image(cls,count=1)
            ret.append(key[0])

        return ret




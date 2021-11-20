#
# This code is based on Vearch Repos code 
# and changes is made to allow calling functions outside vearch
#
import os
import cv2
import base64
import importlib
import urllib.request
import numpy as np
import subprocess

class Base(object):
    pass

class ImageSearch(Base):

    def __init__(self,model_name):
        super(ImageSearch, self).__init__()
        self.model_name = model_name
        self.model_path = f"util.image_extract.{model_name}"
        self.extract_model = self.get_model(self.model_path).load_model()
        # self.detect_model = self.get_model(detect_name).load_model() if detect_name else None
        self.detect_model = None # no detection, May use yolov5 as required after testing

    def pre_process(self, image):
        if self.detect_model:
            bbox = self.detect_model.detect(image)
            image = self.crop(image, bbox)
        return image

    def encode(self, url:str=None, image:np.ndarray=None):
        if image is None:
            try:        
                image = self.read_image(url)
            except Exception as err:
                raise ImageError(f'read {url} failed!')

        image = self.pre_process(image)
        feat = self.extract_model.forward(image)
        assert len(feat) > 0, 'No detect object.'
        return feat[0]


    def get_model(self,model_path):
        """get model by model_path
        Args:
            model_path: the path of model by user define
        Returns:
            the model
        Raises:
            ModuleNotFoundError model not exist
        """
        try:
            model = importlib.import_module(f'{model_path}')
        except ModuleNotFoundError:
            raise ModuleNotFoundError(f'{model_path} is not existed')
        return model

    def read_image(self,imageurl):
        if '.' in imageurl:
            if imageurl.startswith('http'):
                with urllib.request.urlopen(imageurl) as f:
                    resp = f.read()
            elif os.path.exists(imageurl):
                with open(imageurl, 'rb') as f:
                    resp = f.read()
            else:
                raise Exception()
        else:
            resp = base64.b64decode(imageurl)
        image = np.asarray(bytearray(resp), dtype='uint8')
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        return image


    def crop(self, image, bbox):
        if not bbox:
            return image
        x_min, y_min, x_max, y_max = map(int, bbox)
        img_crop = image[y_min:y_max, x_min:x_max]
        return img_crop


    def normlize(self,feat):
        feat = feat/np.linalg.norm(feat)
        return feat.tolist()

    def extrac_feature(self,image):
        if type(image) == str:           
            feature = self.normlize(self.encode(url=image))
        else:
            feature = self.normlize(self.encode(image=image))
        return feature


# def install_package(name):
#     subprocess.run(f'pip install -i https://pypi.tuna.tsinghua.edu.cn/simple {name}', shell=True)


class LoadModelError(Exception):
    pass


class InstallError(Exception):
    pass


class CreateDBAndSpaceError(Exception):
    pass


class ImageError(Exception):
    pass


class HTTPError(Exception):
    pass

if __name__ == '__main__':
    pass

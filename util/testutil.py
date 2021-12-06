from tests.test1 import test_logger
from util.vearchutil import VearchUtil
from pprint import pprint
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from rembg.bg import remove
from PIL import Image
import io

from config.logging import LOGGING_CONF
import logging
import logging.config

logging.config.dictConfig(LOGGING_CONF)
logger = logging.getLogger(__name__)


class TestUtil(object):
    def __init__(self, model_name,data_path = "./data/zerobox") -> None:
        self.model_name = model_name
        self.data_path = data_path
        self.util = VearchUtil(model_name)

    def test(self):
        logger.info(f"TestUtil test {self.model_name}")
        if self.model_name == "alphaalex":
            return self.test_alphaalex()
            
        meta = json.load(open(f"{self.data_path}/meta_test.json", "r"))
        results = list()
        c_class = {"notfound"}
        for item in meta:
            c_class.add(item["class"])
            fname = f"{self.data_path}/images/{item['class']}/{item['file_name']}"

            result = dict()
            result["org_class"] = item["class"]
            result["org_file_name"] = fname
            ret = self.util.search_by_image(image=fname)
            if ret["score"] != -1:
                # found result
                result["test_class"] = ret["data"]["sid"]
                result["test_file_name"] = ret["data"]["image_name"]
            else:
                result["test_class"] = "-1"  # notfound
                result["test_file_name"] = ""

            results.append(result)

        # create consusion matrix
        # go through test image, random pick a class from mapping
        c_class = list(c_class)
        c_class.sort()
        n_class = len(c_class)
        cmatrix = pd.DataFrame(np.zeros((n_class, n_class)))
        cmatrix.columns = c_class
        cmatrix.index = c_class

        wrong_results = []
        for item in results:
            target_label = item["org_class"]
            found_label = item["test_class"]
            cmatrix.loc[target_label, found_label] += 1
            if target_label != found_label:
                wrong_results.append(item["org_file_name"])

        accuracy = np.trace(cmatrix.to_numpy()) / np.sum(cmatrix.to_numpy())

        return (accuracy, cmatrix, wrong_results)

    def test_alphaalex(self):
        logger.info(f"TestUtil test_alphaalex {self.model_name}")

        meta = json.load(open(f"{self.data_path}/meta_test.json", "r"))
        results = list()
        c_class = {"notfound"}
        for item in meta:
            c_class.add(item["class"])
            fname = f"{self.data_path}/images/{item['class']}/{item['file_name']}"

            result = dict()
            result["org_class"] = item["class"]
            result["org_file_name"] = fname
            ret = self.util.search_by_image(image=fname)
            if ret["score"] != -1:
                # found result
                result["test_class"] = ret["data"]["sid"]
                result["test_file_name"] = ret["data"]["image_name"]
            else:
                result["test_class"] = "-1"  # notfound
                result["test_file_name"] = ""

            results.append(result)

        # create consusion matrix
        # go through test image, random pick a class from mapping
        c_class = list(c_class)
        c_class.sort()
        n_class = len(c_class)
        cmatrix = pd.DataFrame(np.zeros((n_class, n_class)))
        cmatrix.columns = c_class
        cmatrix.index = c_class

        wrong_results = []
        for item in results:
            target_label = item["org_class"]
            found_label = item["test_class"]
            cmatrix.loc[target_label, found_label] += 1
            if target_label != found_label:
                wrong_results.append(item["org_file_name"])

        accuracy = np.trace(cmatrix.to_numpy()) / np.sum(cmatrix.to_numpy())

        return (accuracy, cmatrix, wrong_results)

    def plot_images(imgs, row_title=None, **imshow_kwargs):
        if not isinstance(imgs[0], list):
            # Make a 2d grid even if there's just 1 row
            imgs = [imgs]

        num_rows = len(imgs)
        num_cols = len(imgs[0])
        fig, axs = plt.subplots(nrows=num_rows, ncols=num_cols, squeeze=False)
        for row_idx, row in enumerate(imgs):
            for col_idx, img in enumerate(row):
                ax = axs[row_idx, col_idx]
                ax.imshow(np.asarray(img), **imshow_kwargs)
                ax.set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

        if row_title is not None:
            for row_idx in range(num_rows):
                axs[row_idx, 0].set(ylabel=row_title[row_idx])

    def test_color_correction(self,image_name):
        sid = image_name.split('/')[-2]

        model_name = "vgg16"
        util = VearchUtil(model_name=self.model_name)
        item = util.search_by_image(image=image_name)
        # pprint(item)



        image = cv2.imread(image_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image2 = cv2.imread(item["data"]["image_name"])
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)

        if item["data"]["sid"] == sid:
            pprint(f" corect result")
            return (True, [image,image2])

        image3 = cv2.cvtColor(image,cv2.COLOR_RGB2LAB)
        image3 = image3.astype("float32")
        min = np.min(image3[:,:,0])
        max = np.max(image3[:,:,0])
        mean = np.mean(image3[:,:,0])
        
        image3[:,:,0] *= (180/mean)
        image3[image3[:,:,0] > 255,0 ] = 255 

        image3 = image3.astype("uint8")
        image3 = cv2.cvtColor(image3,cv2.COLOR_Lab2BGR)
        item2 = util.search_by_image(image=image3)
        image3 = cv2.cvtColor(image3,cv2.COLOR_BGR2RGB)
        # pprint(item2)

        correct = True if item2["data"]["sid"] == sid else False

        image_f = cv2.imread(item2["data"]["image_name"])
        image_f = cv2.cvtColor(image_f, cv2.COLOR_BGR2RGB)

        return (correct,[image,image3,image_f])


    def remove_bg(self,image_name):
        f = np.fromfile(image_name)
        result = remove(f)
        pil_image = Image.open(io.BytesIO(result)).convert("RGBA")
        # pil_image.save("bg_removed.png")

        # pil_image = PIL.Image.open('image.jpg')
        opencvImage = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGBA2BGR)
        return opencvImage

    
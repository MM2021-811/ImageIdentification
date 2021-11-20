from posixpath import join
from pprint import pprint
import os
from shutil import copyfile
import shutil
import json
import uuid
from util import vearchutil

from config.logging import LOGGING_CONF
import logging
import logging.config

logging.config.dictConfig(LOGGING_CONF)
logger = logging.getLogger(__name__)

# Loading data from Food101 dataset

# step 1 prepare small set of food101
# this will add naming convesion to the class name
# restaurant_foodname
# image name: id.jpg In real application the id should be uuid

def create_food101_subset(num_images = 100):
    """loop through each class and pick 100 images.
    """
    data_path = "./data/food-101"
    dest_path = "./data/mm803"

    classes = os.listdir(f"{data_path}/images")
    meta = []
    for c in classes:
        cnt = 0
        images = os.listdir(os.path.join(data_path,"images",c))
        for fname in images:
            obj = dict()
            cnt += 1
            if cnt <= num_images:
                obj["class"] = c
                obj["restaurant"] = "restauranta"
                obj["id"] = fname[:-4]
                obj["file_name"] = obj["restaurant"] + "_" + fname
            elif cnt > num_images and cnt <= (num_images * 2):
                obj["class"] = c
                obj["restaurant"] = "restaurantb"
                obj["id"] = fname[:-4]
                obj["file_name"] = obj["restaurant"] + "_" + fname
            elif cnt > (num_images * 2):
                break
            meta.append(obj)

    # copy files
    for item in meta:
        src = os.path.join(data_path,"images", item["class"], item["id"]+".jpg")
        folder = os.path.join(dest_path,"images",item["class"])
        if not os.path.exists(folder):
            os.mkdir(folder)

        dest = os.path.join(dest_path,"images",item["class"], item["file_name"])

        shutil.copyfile(src,dest)

    json.dump(meta,open(f"{dest_path}/meta.json","w"),indent=4)

    return

def load_data_to_vearch(data_path="./data/mm803",model_name="vgg16"):
    meta = json.load(open(f"{data_path}/meta.json","r"))
    util = vearchutil.VearchUtil(model_name)
    for item in meta:
        fname = f"{data_path}/images/{item['class']}/{item['file_name']}"
        logger.debug(f"proncessing {fname}")
        restaurant = item["restaurant"]
        # feature = util.extract_feature(fname)

        res = util.add_image_index(fname,sid=item["id"], keyword=restaurant,tags=item["class"])
        logger.debug(res)
        # break

if __name__ == '__main__':
    print("This process will create subset of food101 and index them in vearch")
    print("Donwload Food101 and put under ./data/")
    create_food101_subset()
    load_data_to_vearch()

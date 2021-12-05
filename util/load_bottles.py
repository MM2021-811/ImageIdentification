from posixpath import join
from pprint import pprint
import os
from shutil import copyfile
import shutil
import json
import uuid
from util import vearchutil
import math
import argparse
import numpy as np
from rembg.bg import remove
from PIL import Image
import io

from config.logging import LOGGING_CONF
import logging
import logging.config

logging.config.dictConfig(LOGGING_CONF)
logger = logging.getLogger(__name__)

# Loading data 
# datset structure
# dataset_name
#      class_name
#              image_name

# naming convesion:
#   folder_name == class_name == sid

def create_dataset_metadata(data_path="./data/zerobox",num_images = 100, test_percent=0.2):
    """loop through each class and pick 100 images.
    """
    logger.info(f"create_dataset_metadata({num_images})")
    classes = os.listdir(f"{data_path}/images")
    meta_train = []
    meta_test = []
    for c in classes:
        cnt = 0
        meta = []
        images = os.listdir(os.path.join(data_path,"images",c))
        for fname in images:
            cnt += 1
            obj = dict()            
            obj["class"] = c
            obj["id"] = c
            obj["keyword"] = ""
            obj["file_name"] = fname
            if cnt > num_images:
                break
            meta.append(obj)

        #split to index and test
        test_idx = -1 * math.ceil(cnt * test_percent)
        meta_train.extend(meta[:test_idx])
        meta_test.extend(meta[test_idx:])
    
    meta_all = []
    meta_all.extend(meta_train)
    meta_all.extend(meta_test)
    json.dump(meta_train,open(f"{data_path}/meta_train.json","w"),indent=4)
    json.dump(meta_test,open(f"{data_path}/meta_test.json","w"),indent=4)
    json.dump(meta_all,open(f"{data_path}/meta_all.json","w"),indent=4)
    
    return

def load_data_to_vearch(data_path="./data/zerobox",model_name="vgg16"):
    logger.info(f"load_data_to_vearch({data_path},{model_name})")

    meta = json.load(open(f"{data_path}/meta_train.json","r"))
    util = vearchutil.VearchUtil(model_name)
    for item in meta:
        fname = f"{data_path}/images/{item['class']}/{item['file_name']}"
        logger.debug(f"proncessing {fname}")
        res = util.add_image_index(fname,sid=item["id"], keyword=item["keyword"],tags=item["class"])
        logger.debug(res)
        # break

def create_nobg_dataset(src_path= "./data/zerobox", data_path="./data/zerobox_nobg"):
    logger.info(f"create_nobg_dataset({src_path}, {data_path})")

    meta = json.load(open(f"{src_path}/meta_train.json","r"))
    meta.extend(json.load(open(f"{src_path}/meta_test.json","r")))
    for item in meta:
        fname = f"{src_path}/images/{item['class']}/{item['file_name']}"
        logger.debug(f"proncessing {fname}")
        f = np.fromfile(fname)
        result = remove(f)
        pil_image = Image.open(io.BytesIO(result)).convert("RGBA")

        if not os.path.exists(f"{data_path}/images/{item['class']}"):
            os.mkdir(f"{data_path}/images/{item['class']}")

        dest_name = f"{data_path}/images/{item['class']}/{item['file_name']}"
        pil_image.save(dest_name)

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='Loading data into vearch')
    parser.add_argument('--data-path', type=str, default="./data/zerobox", metavar='P',
                        help='dataset path (default: ./data/zerobox)')
    parser.add_argument('--model-name', type=str, default="vgg16", metavar='M',
                        help='model name(default: vgg16)')
    parser.add_argument('--create-meta', action='store_true', default=False,
                        help='recreate meta file')

    args = parser.parse_args()
    create_meta = args.create_meta
    if create_meta:
        create_dataset_metadata()
    
    load_data_to_vearch(data_path=args.data_path,model_name=args.model_name)

if __name__ == '__main__':
    print("This process will create metadata and index them in vearch")
    main()

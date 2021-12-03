from tests.test1 import test_logger
from util.vearchutil import VearchUtil
from util.testutil import TestUtil
from pprint import pprint
from util.load_bottles import load_data_to_vearch
from init_vearch_local import create_space,delete_space, create_db
from util.trainingutil import AlphaBgTransform
import cv2

# create_food101_subset()
# load_data_to_vearch()

# util = VearchUtil(model_name="vit16")

# image_name="./data/zerobox/images/blackbottles/00001.png"
# ret = util.add_image_index(image_name=image_name,sid="b01")
# pprint(ret)


# image_name="./data/zerobox/images/blackbottles/000034.png"
# item = util.search_by_image(image=image_name)
# pprint(item)

def test_vit16():
    testutil = TestUtil(model_name="vit16")
    (accuracy, cmatrix,wrong_results) = testutil.test()

    pprint(cmatrix)
    pprint(f"Final accuracy: {accuracy}")
    print("Wrong results:")
    pprint(wrong_results)

def test_load():
    load_data_to_vearch(model_name="vgg16")

def load_net(model_name = "alexnet",data_path="./data/zerobox"):
    db_name="bottle"
    create_db(db_name)

    image_name = "./data/zerobox/images/blackbottles/00001.png"
    util = VearchUtil(model_name=model_name)
    item = util.extract_feature(image=image_name)
    feature_dim = len(item)
    print(feature_dim)

    delete_space("bottle",model_name)
    create_space("bottle",model_name,feature_dim=feature_dim,partition=4)
    load_data_to_vearch(data_path=data_path,model_name=model_name)

def eval_net(model_name = "alexnet",data_path = "./data/zerobox"):
    testutil = TestUtil(model_name,data_path)
    (accuracy, cmatrix,wrong_results) = testutil.test()

    pprint(cmatrix)
    pprint(f"Final accuracy: {accuracy}")
    print("Wrong results:")
    pprint(wrong_results)


def test_transform(image_name):
    image = cv2.imread(image_name,cv2.IMREAD_UNCHANGED)

    # transform = AlphaBgTransform()
    img = AlphaBgTransform.to_square(image)
    pprint(img.shape)
    return img


test_transform(image_name= './data/zerobox_nobg/images/white02/output0064.png')

# load_net("vgg16",data_path="./data/zerobox_light")
# eval_net("vgg16",data_path="./data/zerobox_light")
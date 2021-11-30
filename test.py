from tests.test1 import test_logger
from util.vearchutil import VearchUtil
from util.testutil import TestUtil
from pprint import pprint
from util.load_bottles import load_data_to_vearch
from init_vearch_local import create_space,delete_space

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

def test_alexnet():
    delete_space("bottle","alexnet")
    create_space("bottle","alexnet",feature_dim=256)
    load_data_to_vearch(model_name="alexnet")

test_alexnet()
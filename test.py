from tests.test1 import test_logger
from util.vearchutil import VearchUtil
from util.testutil import TestUtil
from pprint import pprint
from util.load_bottles import load_data_to_vearch
from init_vearch_local import create_space,delete_space, create_db

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

def test_net(model_name = "alexnet"):
    db_name="bottle"
    create_db(db_name)

    image_name = "./data/zerobox/images/blackbottles/00001.png"
    util = VearchUtil(model_name=model_name)
    item = util.extract_feature(image=image_name)
    feature_dim = len(item)
    print(feature_dim)

    delete_space("bottle",model_name)
    create_space("bottle",model_name,feature_dim=feature_dim,partition=4)
    load_data_to_vearch(model_name=model_name)

def eval_net(model_name = "alexnet"):
    testutil = TestUtil(model_name)
    (accuracy, cmatrix,wrong_results) = testutil.test()

    pprint(cmatrix)
    pprint(f"Final accuracy: {accuracy}")
    print("Wrong results:")
    pprint(wrong_results)

# test_net("vgg16")
eval_net("vgg16")
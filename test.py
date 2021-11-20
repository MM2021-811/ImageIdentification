from tests.test1 import test_logger
from util.vearchutil import VearchUtil
from util.testutil import TestUtil
from pprint import pprint

# create_food101_subset()
# load_data_to_vearch()

util = VearchUtil(model_name="vgg16")

# image_name="./data/zerobox/images/blackbottles/00001.png"
# ret = util.add_image_index(image_name=image_name,sid="b01")
# pprint(ret)


# image_name="./data/zerobox/images/blackbottles/000034.png"
# item = util.search_by_image(image=image_name)
# pprint(item)

testutil = TestUtil(model_name="vgg16")
(cmatrix,wrong_results) = testutil.test()

pprint(cmatrix)

print("Wrong results:")
pprint(wrong_results)

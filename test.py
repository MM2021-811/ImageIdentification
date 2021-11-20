from tests.test1 import test_logger
from util.vearchutil import VearchUtil
from pprint import pprint

# create_food101_subset()
# load_data_to_vearch()

util = VearchUtil(model_name="vgg16")
image_name="./data/zerobox/images/blackbottles/000034.png"
item = util.search_by_image(image=image_name)
pprint(item)
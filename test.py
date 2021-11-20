from tests.test1 import test_logger
from util.load_f101 import create_food101_subset, load_data_to_vearch
from util.vearchutil import VearchUtil
from pprint import pprint

# create_food101_subset()
# load_data_to_vearch()

util = VearchUtil(model_name="vgg16")
image_name="./data/mm803/images/spaghetti_bolognese/restaurantb_54586.jpg"
restaurant = "restaurantb"
item = util.search_by_image(keyword=restaurant,image_name=image_name)
pprint(item)
import json
from json.decoder import JSONDecodeError
import os
from datetime import datetime
from typing import List
from uuid import uuid4
import requests
from config import settings
from util.image_extract.vearch import ImageSearch

from config.logging import LOGGING_CONF
import logging
import logging.config
import time
logging.config.dictConfig(LOGGING_CONF)
logger = logging.getLogger(__name__)
import http.client
http.client._MAXLINE = 655360

os.environ['CUDA_VISIBLE_DEVICES'] ='4'


class ParameterError(Exception):
    pass


class VearchApiError(Exception):
    pass

class VearchIndexError(Exception):
    pass

class VearchUtil:
    def __init__(self, model_name="vgg16") -> None:
        super().__init__()
        self.server_url = settings.VEARCH_URL
        self.header = {"Content-Type": "application/json"}
        self.vearchutil = ImageSearch(model_name)
        self.model_name = model_name
        self.db_name = "bottle"
        self.space_name = model_name

        # self.header = {"Authorization": self.token}
        # self.token = "Token 8d8cb171e59b5fb5c83fa074c1a97e47fef44d64"

    def extract_feature(self, image):
        return self.vearchutil.extrac_feature(image)

    def add_image_index(self, image_name: str, sid: str, keyword: str, tags: list = [], uuid: str = None):
        """Index new images

        Args:
            image_name (str): image_name, can be full path.
            keyword (str): keyword for searching.
                e.g. use Restaurant name as a keyword for search and isolate
                the results only belongs to this restaurant
            tags (list): e.g. food name

        Returns:
            Json: Vearch response
        """
        if uuid is None:
            uuid = uuid4().__str__()

        data = {
            "image_name": image_name,
            "image": {"feature": self.extract_feature(image_name)},
            "model_name": self.vearchutil.model_name,
            "keyword": keyword,
            "uuid": uuid,
            "sid": sid,
            "tags": tags,
        }

        # logger.debug(data)
        # json.dump(data,open("./data.json",'w'))
        # data1 = json.load(open("./data.json","r"))
        # logger.debug(data1)

        url = f"{self.server_url}/{self.db_name}/{self.space_name}/{uuid}"
        # logger.debug(url)
        response = requests.post(url, json=data, headers=self.header)

        # remove unecessary files
        # os.remove(f"{settings.LOCAL_IMAGE_PATH}/{image_name}")

        logger.debug(response.status_code)
        if response.status_code != 200:
            logger.error(response.text)
            raise VearchIndexError(response.text)


        return response.text

    def search_by_image(self, keyword: str = None, image = None, feature: list = None, return_records:int = 1) -> dict:
        """[summary]            
        Args:
            image ([type]): can be str: image file name or numpy.ndarray returned by cv2.read(image)

        Returns:
            [dict]: [description]
        """

        if image is None and feature is None:
            raise ParameterError(
                f"image and feature can't be None at the same time")

        if feature is None:
            feature = self.extract_feature(image)

        # have to use is_brute_search = 1
        if keyword is not None:
            # have to use is_brute_search = 1
            data = {
                "query": {
                    "filter": [
                        {
                            "term": {
                                "operator": "and",
                                "keyword": [keyword]
                            }
                        },
                        {
                            "term": {
                                "operator": "and",
                                "model_name": [self.vearchutil.model_name]
                            }
                        }
                    ],
                    "sum": [
                        {
                            "feature": feature,
                            "field": "image"
                        }
                    ]
                },
                "is_brute_search": 1
            }
        else:
            data = {
                "query": {
                    "filter": [
                        {
                            "term": {
                                "operator": "and",
                                "model_name": [self.vearchutil.model_name]
                            }
                        }
                    ],
                    "sum": [
                        {
                            "feature": feature,
                            "field": "image"
                        }
                    ]
                },
                "is_brute_search": 1
            }

        s1 = json.dumps(data)

        url = f"{self.server_url}/{self.db_name}/{self.space_name}/_search?size=10"
        # vearch api has limitation, must pass in as string other than dict
        response = requests.post(url, data=s1, headers=self.header)

        logger.debug(response.status_code)
        if response.status_code != 200:
            raise VearchApiError(response.text)

        try:
            data = json.loads(response.text)
        except JSONDecodeError as ex:
            data = json.loads(response.text + "}")


        found_total = data.get("hits").get("total",0)
        if(found_total > 0):
            hits = data.get("hits").get("hits")
            cnt = min(found_total,return_records)
            items = []
            for i in range(cnt):
                f_hit = hits[0]
                item = dict()
                item["score"] = f_hit.get("_score")
                item["vearch_id"] = f_hit.get("_id")
                item["data"] = f_hit.get("_source")
                items.append(item)
            
            if return_records == 1:
                return items[0]
            else:
                return items
        else:
            item = item = dict()
            item["score"] = -1
            item["vearch_id"] = None
            item["data"] = ""
            return item

    def delte_image_index(self, uuid: str):
        """Remove one image index   
           To remove all drop the space and recreate it.
        Args:
            uuid (str): 

        Returns:
            api reponse.text
        """
        url = f"{self.server_url}/{self.db_name}/{self.space_name}/{uuid}"
        response = requests.delete(url, headers=self.header)

        logger.debug(response.status_code)

        if response.status_code != 200:
            logger.error(response.text)

        return response.text

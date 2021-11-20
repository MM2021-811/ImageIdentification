import json
import requests
from util.load_f101 import create_food101_subset, load_data_to_vearch

# SERVER_URL = "http://vearch_plugin:4101"
SERVER_URL = "http://localhost:4101"


def init_vearch():
    # self.token = "Token cb6f6b82bcd37e02ecacb16dfdb0be3e3ae6fa68"
    # self.server_url = "http://localhost:8000/api/foods"
    # self.header = {"Authorization": self.token}

    # create db
    header = {"Content-Type": "application/json"}
    url = f"{SERVER_URL}/db/_create"
    data = {"name": "foods"}
    response = requests.put(url, json=data, headers=header)
    print(response.text)

    # create space
    url = f"{SERVER_URL}/space/foods/_create"
    data = {
        "name": "foods",
        "partition_num": 1,
        "replica_num": 1,
        "engine": {
            "name": "gamma",
            "index_size": 70000,
            "max_size": 10000000,
            "id_type": "String",
            "retrieval_type": "IVFPQ",
            "retrieval_param": {
                "metric_type": "InnerProduct",
                "ncentroids": 256,
                "nsubvector": 32,
            },
        },
        "properties": {
            "image_name": {"type": "keyword", "index": True},
            "image": {"type": "vector", "dimension": 512, "format": "normalization"},
            "model_name": {"type": "keyword", "index": True},
            "keyword": {"type": "keyword", "index": True},
            "tags": {"type": "string", "array": True, "index": True},
            "uuid": {"type": "keyword", "index": True},
            "sid": {"type": "keyword", "index": True},
        },
    }
    response = requests.put(url, json=data, headers=header)
    print(response.content)

    # verify be list details
    url = f"{SERVER_URL}/space/foods/foods"
    data = {"name": "foods"}
    response = requests.get(url, json=data, headers=header)
    print(response.content)


init_vearch()

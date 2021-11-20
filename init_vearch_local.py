import json
import requests
from config.settings import VEARCH_URL

# SERVER_URL = "http://vearch_plugin:4101"
SERVER_URL = VEARCH_URL


def init_vearch():
    # create db
    header = {"Content-Type": "application/json"}
    url = f"{SERVER_URL}/db/_create"
    data = {"name": "bottle"}
    response = requests.put(url, json=data, headers=header)
    print(response.text)

    # create space
    url = f"{SERVER_URL}/space/bottle/_create"
    data = {
        "name": "bottle",
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
    url = f"{SERVER_URL}/space/bottle/bottle"
    data = {"name": "bottle"}
    response = requests.get(url, json=data, headers=header)
    print(response.content)


init_vearch()

import json
import requests
from config.settings import VEARCH_URL

# SERVER_URL = "http://vearch_plugin:4101"
SERVER_URL = VEARCH_URL

def create_db(db_name:str = "foods"):
    # create db
    header = {"Content-Type": "application/json"}
    url = f"{SERVER_URL}/db/_create"
    data = {"name": db_name}
    response = requests.put(url, json=data, headers=header)
    print(response.text)

def delete_space(db_name,space_name):
    header = {"Content-Type": "application/json"}
    url = f"{SERVER_URL}/space/{db_name}/{space_name}"
    data = {"name": space_name}
    response = requests.delete(url, json=data, headers=header)
    print(response.text)
    return response

def create_space(db_name,space_name,feature_dim=512,partition=4):
    header = {"Content-Type": "application/json"}
    url = f"{SERVER_URL}/space/{db_name}/_create"
    data = {
        "name": space_name,
        "partition_num": partition,
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
            "image": {"type": "vector", "dimension": feature_dim, "format": "normalization"},
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
    url = f"{SERVER_URL}/space/{db_name}/{space_name}"
    data = {"name": space_name}
    response = requests.get(url, json=data, headers=header)
    print(response.content)


def init_vearch():
    # self.token = "Token cb6f6b82bcd37e02ecacb16dfdb0be3e3ae6fa68"
    # self.server_url = "http://localhost:8000/api/foods"
    # self.header = {"Authorization": self.token}
    db_name="bottle"
    # create db
    create_db(db_name)
    # create space for vgg16
    create_space(db_name, space_name="vgg16",feature_dim=512,partition=4)

    # alexnet 256


if __name__ == "__main__":
    init_vearch()

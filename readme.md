# Nearly identical Image Indentification
This repository is created for MM program at UofA for student projects. 

## license
The license for this repo is under MIT license. The dependable libraries and packages' license please check their license yourself.

1. Vearch (Apache 2): https://github.com/vearch/vearch/blob/master/LICENSE

## Acredit
1. This repository uses [vearch](https://github.com/vearch/vearch) and [python-algorithm-plugin](https://github.com/vearch/python-algorithm-plugin) 
2. [albumentations](https://github.com/albumentations-team/albumentations)
```
@Article{info11020125,
    AUTHOR = {Buslaev, Alexander and Iglovikov, Vladimir I. and Khvedchenya, Eugene and Parinov, Alex and Druzhinin, Mikhail and Kalinin, Alexandr A.},
    TITLE = {Albumentations: Fast and Flexible Image Augmentations},
    JOURNAL = {Information},
    VOLUME = {11},
    YEAR = {2020},
    NUMBER = {2},
    ARTICLE-NUMBER = {125},
    URL = {https://www.mdpi.com/2078-2489/11/2/125},
    ISSN = {2078-2489},
    DOI = {10.3390/info11020125}
}
```


# Installation
All commands list here are sample commands. And this is based on Linux(Ubuntu). If you are running on other OS, please issue the command accordingly to your OS.

## Prequirements

* Python 3.8
* Clone this project and create your own branch and work on it. `checkout -b yourname`

## Setup Python environment
* setup python virtual environment `python -m venv .venv`
* activate virtual env `. .venv/bin/activate`
* upgrade pip `pip install --upgrade pip`
* Install pytorch. Please follow https://pytorch.org/get-started/locally/  instructions to install pytorch base on your machine model.
* install other required package `pip install -r requirements.txt`
* 

## Bring up Vearch
If you need change verach plugin port, you can find the settings in `docker-compose.yml`. If you do make the change, you also need update VEARCH_URL in `config/settings.py`

```
docker-compose pull

docker-compose up -d

# tear down.  use -v and remove all data when you needs.
# docker-compose down -v

```

## Inititalized data

```bash
./init.sh
```

## Prepare dataset
1. download [zeroboxbottles](https://data.vision.ee.ethz.ch/cvl/datasets_extra/food-101/) dataset
2. unzip the dataset to ./data. After zip the folder should be `./data/zerobx/`
   
## Index subset image
run `python util/load_bottles.py`. This script will load subset and index them in vearch. 
This step may requires to run based on the test cases. When using different model(vgg, resnet and etc) must index them with same model

## Test different model
Highlevel steps
1. create new model and put under `./util/image_extract/`
2. call `load_data_to_vearch(data_path,model_name)`
3. test searching result and record the result for later comparision 


# Quick Notes
## Vearch

### Vearch search sytax
```json
{
  "query": {
      "filter": [
          {
        "term":{
          "operator":"and",
          "keyword":["somekeyword"] 
        }
      }  
      ],
    "sum": [
      {
        "feature": [
                0.015417137,
                0.06690312,
                0.027473053,
                ...                
                0.03291337,
                0.009573658,
                0.029383807
            ],
        "field": "image"        
      }            
    ]
  },
  "is_brute_search": 1
}
```
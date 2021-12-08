# Nearly identical Image Indentification
This repository is created for MM program at UofA for student projects. 

## license
The license for this repo is under MIT license. Most of dependable libraries and packages' license is MIT based. Please check their license yourself before reusing the code in this repo.

1. Vearch (Apache 2): https://github.com/vearch/vearch/blob/master/LICENSE

## Acredit
1. Vearch 
  [vearch](https://github.com/vearch/vearch)
  [python-algorithm-plugin](https://github.com/vearch/python-algorithm-plugin)
  [paper](https://arxiv.org/abs/1908.07389)
   ```
   @unknown{unknown,
    author = {Li, Jie and Liu, Haifeng and Gui, Chuanghua and Chen, Jianyu and Ni, Zhenyun and Wang, Ning},
    year = {2019},
    month = {08},
    pages = {},
    title = {The Design and Implementation of a Real Time Visual Search System on JD E-commerce Platform}
    }
   ```

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
3. Low-light-Image-Enhancement by 
   [Paper](https://arxiv.org/pdf/1910.13688v1.pdf)
   [Code](https://github.com/pvnieo/Low-light-Image-Enhancement)
   ```
   @article{article,
    author = {Zhang, Qing and Nie, Yongwei and Zheng, Wei‐Shi},
    year = {2019},
    month = {10},
    pages = {243-252},
    title = {Dual Illumination Estimation for Robust Exposure Correction},
    volume = {38},
    journal = {Computer Graphics Forum},
    doi = {10.1111/cgf.13833}
    }
    ```
4. Alexnet Implementation
   Alexnet code is based on the source code from this [article](https://medium.com/analytics-vidhya/alexnet-a-simple-implementation-using-pytorch-30c14e8b6db2) by Toluwani Aremu 2021. 

5. LA-Transformer
  [paper](https://www.researchgate.net/publication/352209565_Person_Re-Identification_with_a_Locally_Aware_Transformer)
  [code](https://github.com/SiddhantKapil/LA-Transformer)
  ```
  @unknown{unknown,
  author = {Sharma, Charu and Kapil, Siddhant and Chapman, David},
  year = {2021},
  month = {06},
  pages = {},
  title = {Person Re-Identification with a Locally Aware Transformer}
  }
  ```

6. Rembg
  This work is built on top of U2Net.
  [Library code](https://github.com/danielgatis/rembg)
  [U2Net](https://github.com/xuebinqin/U-2-Net)

7. Differnet
  [code](https://github.com/marco-rudolph/differnet)
  [paper](https://arxiv.org/abs/2008.12577)
  ```
  @unknown{unknown,
    author = {Rudolph, Marco and Wandt, Bastian and Rosenhahn, Bodo},
    year = {2020},
    month = {08},
    pages = {},
    title = {Same Same But DifferNet: Semi-Supervised Defect Detection with Normalizing Flows}
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
1. download [zeroboxbottles](https://drive.google.com/file/d/1r-w3dznBSpqB83UJz9OvjHTk6cdUq-b0/view?usp=sharing) dataset
2. unzip the dataset to ./data. After zip the folder should be `./data/zerobx/`
   
## Index subset image
run `python util/load_bottles.py`. This script will load images and index them in vearch. 
This step may requires to run based on the test cases. When using different model(vgg, resnet and etc) must index them with same model

## Test different model
The test images are listed in meta_test.json. 

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
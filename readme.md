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
  8. Training scripts
    The scripts train_xxxxx.py is based on [this toturial](https://www.analyticsvidhya.com/blog/2019/10/how-to-master-transfer-learning-using-pytorch/). Author: Pulkit Sharma Year: 2019



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
## Prepare dataset and pre-trained model
1. download [dataset_models.tar.gz](https://drive.google.com/file/d/1fDzs5_Ixxs_QeI2Oo5VmG7dJZMVaov9n/view?usp=sharing) dataset
2. unzip the dataset and models to current folder. `tar -xvf dataset_models.tar.gz`


# Our work and main notebook
The main notebook is `report_demo.ipynb`. Please reference the file for details of this project work. If you are reading the file from github directly some images is not shown due to some issues from Github. Please read `report_demo.pdf` instead.

## Our major codes

1. Alexnet_xxxx.ipynb  
  Alexnet related training notebook
2. docker-compose.yml  
  The Yaml file which control how to create containers for VEARCH
3. retport_demo.ipynb or report_demo.pdf 
  The summary of the project and it also includes entry of the code of creating image index, test each feature extration models, results of testing and other work related to this project.
4. train_alphaalexnet.py  traning script for AlphaAlexNet.
  train_siamesealexnet.py  training script for SiameseAlexNet. Please referece to our final report for these networks.
5. util/trainingutil.py 
   training relateded code. It includes the model we defined, such as AlphaAlexNet, SiameseAlexnet, cutomized Transforms class and etc.
   
6. util/vearchutil
   libray code we wrote for calling Vearch API

7. util/testutil
    reuable libray code for testing purpose. 

8. util/alphaalex.py
    The code for extracting features into vector.


  
  




# File and structure in this project
```
.
├── Alexnet_2bottle.ipynb
├── Alexnet_2classes.ipynb
├── Alexnet_4bottle.ipynb
├── Alexnet_4bottle_unbiased.ipynb
├── alexnet_training.ipynb
├── alphaalexnet_training.ipynb
├── bg_removed.png
├── color_corrected.png
├── colorspace.ipynb
├── compare_models.ipynb
├── config
│   ├── __init__.py
│   ├── logging.py
│   ├── __pycache__
│   ├── settings.py
│   └── vearch_config.py
├── data
│   ├── zerobox
│   ├── zerobox_light
│   └── zerobox_nobg
├── docker-compose.yml
├── enhanced_bottle.png
├── images
│   ├── architecture.png
│   ├── Siamese.png
│   └── workflow.png
├── index_images.sh
├── init.sh
├── init_vearch_local.py
├── logs
│   ├── app.log
│   └── readme.md
├── maske.png
├── models
│   ├── alexnet_alpha.pth
│   ├── alexnet_org.pth
│   ├── bottle_siamese.pth
│   └── bottle_siamese_tmp.pth
├── readme.md
├── report_demo.ipynb
├── requirements.txt
├── test.png
├── test.py
├── train_alexnet.py
├── train_alphaalexnet.py
├── train_siamese_alexnet.py
├── train_siamese_vgg16net.py
├── util
│   ├── exposure_enhancement.py
│   ├── huggingface.ipynb
│   ├── image_extract
│   ├── __init__.py
│   ├── latransformer_util.py
│   ├── load_bottles.py
│   ├── __pycache__
│   ├── testutil.py
│   ├── trainingutil.py
│   └── vearchutil.py
├── vearch
│   ├── config_plugin.py
│   ├── config.toml
│   └── Dockerfile
├── vearch_plugin
│   ├── bin
│   ├── Dockerfile
│   ├── Dockerfile.NonChina
│   ├── images
│   ├── __init__.py
│   ├── README.md
│   └── src
├── vgg16.ipynb
```
## Some script and functions
utl/trainingutil.py  
Models of AlphaAlexNet, SiameseAlexNet

utl/testutil.py 

testing and benchmark utilities

Alexnet_*.ipynb Alexnet related scripts.

vearch/ 
Vearch code(third party)

vearch_plugin/ 
vearch plugin (third party)



## Index subset image
Please reference `report_demo.ipynb` as examples.


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
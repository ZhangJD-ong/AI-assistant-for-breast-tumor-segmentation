# AI-assistant-for-breast-tumor-segmentation

## Introduction:
This project includes both train/test code for training models on uses' own data or fine-tuning models.


## Requirements:
* python 3.10
* pytorch 1.12.1
* numpy 1.23.3
* tensorboard 2.10.1
* simpleitk 2.1.1.1
* scipy 1.9.1

## Setup

### Installation
Clone and repo and install required packages:
```
git clone git@github.com:ZhangJD-ong/AI-assistant-for-breast-tumor-segmentation.git
pip install -r requirements.txt
```
### Dataset
* For training the segmentation models, you need to put the data in this format：

```
./data
├─train.txt
├─test.txt
├─Guangdong
      ├─Guangdong_1
          ├─P0.nii.gz
          ├─P1.nii.gz
          ├─P2.nii.gz
          ├─P3.nii.gz
          ├─P4.nii.gz     
          └─P5.nii.gz
      ├─Guangdong_2
      ├─Guangdong_3
      ...
├─Guangdong_breast
      ├─Guangdong_1.nii.gz
      ├─Guangdong_2.nii.gz
      ├─Guangdong_2.nii.gz
      ...
├─Guangdong_gt
      ├─Guangdong_1.nii.gz
      ├─Guangdong_2.nii.gz
      ├─Guangdong_2.nii.gz
      ...         
└─Yunzhong
└─Yunzhong_breast
└─Yunzhong_gt
└─Ruijin
└─Ruijin_breast
└─Ruijin_gt
...
```
* The format of the train.txt / test.txt is as follow：
```
./data/train.txt
├─'Guangdong_1'
├─'Guangdong_2'
├─'Guangdong_3'
...
├─'Yunzhong_100'
├─'Yunzhong_101'
...
├─'Ruijin_1010'
...
```
* For inference on own data, user should put the new data in this format:
```
./Inference-code./Data./Original_data
├─name1
      ├─P0.nii.gz
      ├─P1.nii.gz
      ...
      └─P5.nii.gz
├─name2
├─name3
...
```










# AI-assistant-for-breast-tumor-segmentation

## Paper:
Please see: A robust and efficient AI assistant for breast tumor segmentation from DCE-MRI via a spatial-temporal framework
https://www.cell.com/patterns/fulltext/S2666-3899(23)00195-2

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
pip install -r requirement.txt
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
./Inference-code/Data/Original_data
├─name1
      ├─P0.nii.gz
      ├─P1.nii.gz
      ...
      └─P5.nii.gz
├─name2
├─name3
...
```

### Training and testing
* For training the segmentation model, please add data path and adjust model parameters in the file: ./Train-and-test-code/options/BasicOptions.py. 
```
cd ./Train-and-test-code
python train.py
python test.py
```
### Inference on own data
* Please put the new data in the fold: ./Inference-code/Data/Original_data. The segmentation results can be find in ./Inference-code/Results/Tumor/.
```
cd ./Inference-code
python test.py
```
* We release the well-trained model (Can be downloaded from https://drive.google.com/drive/folders/1Sos8NK4zzkT1L96saffsg4EpUyjwRSjm?usp=sharing , due to the memory limitation in Github) and five samples to guide usage. Please put the download 'Trained_model' folder in ./Inference-code/.
* The data can only be used for academic research usage.
* More data are available at https://doi.org/10.5281/zenodo.8068383.

## Citation
If you find the code or data useful, please consider citing it using the following format:

* Zhang et al., A robust and efficient AI assistant for breast tumor segmentation from DCE-MRI via a spatial-temporal framework, Patterns (2023), https://doi.org/10.1016/j.patter.2023.100826







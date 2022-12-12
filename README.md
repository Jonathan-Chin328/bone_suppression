# Bone suppression
Input cxr images and output the images without bone. 
- Reference: [Chest X‐Ray Bone Suppression for Improving Classification of
Tuberculosis‐Consistent Findings](https://arxiv.org/abs/2104.04518)
- Implement ResNet-BS model

## 1. Overview

- [**config/**](config) : config file for train and inference
- [**train.py**](train.py) : python script for excuting taining
- [**inference.py**](inference.py) : python script for excuting inference
- [**transfer.py**](transfer.py) : python script for transform traning data from kaggle

## 2. Train 
```
python train.py --config ./config/train.yaml
```
```train.yaml```
| type        | name              | description                                                                             |
|:------------|:------------------|:----------------------------------------------------------------------------------------|
| path        | dataset           | where the dataset is (have both source and target)                                      |
|             | save_path         | where to save model and log.txt                                                         |
| parameter   | train_set_ratio   | the ratio to split dataset into training and validation                                 |
|             | epoch             | epoch to train                                                                          |
| model       | num_filters       | how many filter for the convolution layer                                               |
|             | num_res_blocks    | how many resnet block                                                                   |
|             | res_block_scaling | scaling factor to sacle down the residuals before adding back to the convolutional path |
> Due to limit gpu space:  
> Train on img size 256, num_filters = 16, batch_size = 8   
> Train on img size 1024, num_filters = 11, batch_size = 1

## 3. Inference 
```
python inference.py --config ./config/inference.yaml
```
```inference.yaml```
| type        | name              | description                                                                             |
|:------------|:------------------|:----------------------------------------------------------------------------------------|
| path        | dataset           | the path of data which want to have back bone suppression inference                     |
|             | load_path         | where to load well trained model                                                        |
|             | save_path         | where to save inference data                                                            |
| model       | num_filters       | how many filter for the convolution layer                                               |
|             | num_res_blocks    | how many resnet block                                                                   |
|             | res_block_scaling | scaling factor to sacle down the residuals before adding back to the convolutional path |

## 4. Transfer
- To have better performance on cxr dataset, preprocess the JSRT dataset from kaggle (See [**transfer.py**](transfer.py) to know more detail)
```
python transfer.py
```
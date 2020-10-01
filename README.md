[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center">Apollo</h1>
<h5 align="center">Apollo: An Adaptive Parameter-wise DIAgonal Quasi-Newton Method for Nonconvex Stochastic Optimization</h5>

This is the Pytorch implementation for [Apollo: An Adaptive Parameter-wise DIAgonal Quasi-Newton Method for Nonconvex Stochastic Optimization](https://arxiv.org/abs/2009.13586)

## Requirements
* Python >= 3.6
* Pytorch >= 1.5.0
* apex
* lmdb >= 0.94
* overrides 
* tqdm


## Installation
1. Install [NVIDIA-apex](https://github.com/NVIDIA/apex).
2. Install [Pytorch and torchvision](https://pytorch.org/get-started/locally/)

## Data
1. One Billion Words can be obtained [here](https://www.statmt.org/lm-benchmark/)
2. WMT'14 English to German (EN-DE) can be obtained with scripts provided in [fairseq](https://github.com/pytorch/fairseq/blob/master/examples/translation/README.md#wmt14-english-to-german-convolutional).

## Experimental Results

### Image Classification
<img src="./docs/images/classify.png" width="1000"/>

| Method     |  CIFAR-10 (%)      |  CIFAR-10 (%)      |  ImageNet (%)      |  ImageNet (%)      |
| :--------- | :----------------: | :----------------: | :----------------: | :----------------: |
|            |  **milestone**     |  **cosine**        |  **milestone**     |  **cosine**        |
| SGD        |  93.91 (0.07)      |  94.53 (0.27)      |  77.19 (0.07)      |  78.17 (0.06)      |
| Adam       |  91.41 (0.30)      |  91.56 (0.19)      |  71.72 (0.13)      |  71.19 (0.10)      |
| RAdam      |  91.80 (0.04)      |  91.88 (0.15)      |  72.37 (0.08)      |  71.64 (0.14)      |
| Adam-adj   |  93.74 (0.15)      |  94.24 (0.09)      |  76.86 (0.06)      |  77.54 (0.16)      |
| RAdam-adj  |  93.88 (0.11)      |  94.38 (0.25)      |  76.91 (0.07)      |  77.68 (0.08)      |
| **Apollo** |  **94.20 (0.12)**  |  **94.60 (0.06)**  |  **77.90 (0.06)**  |  **78.54 (0.09)**  |

For the model training of image classification, please go this [folder](https://github.com/XuezheMax/apollo/tree/master/classification)


[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

<h1 align="center">Apollo</h1>
<h5 align="center">Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization</h5>

This is the Pytorch implementation for [Apollo: An Adaptive Parameter-wise Diagonal Quasi-Newton Method for Nonconvex Stochastic Optimization](https://arxiv.org/abs/2009.13586)

## Table of Contents

- [Requirements](#requirements)
- [Installation](#installation)
- [Notes](#notes)
- [Experiments](#experimental-results)
- [Discussion](#discussion)

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

## Notes
- The ranges of hyper parameters of Apollo, such as learning rate, are very different with Adam (see the appendix of the paper for details). To apply Apollo to your tasks, a reasonable set of hyper parameters to begin with is ```lr=1.0, eps=1e-4, init_lr=0.01, warmup=100```.
- Warmup plays a super important role for Apollo. Please set warmup to at least `100` updates to achieve stable convergence.

## Experimental Results

### Image Classification
<img src="./docs/images/classify_full.png" width="1000"/>

| Method     |  CIFAR-10 (%)      |  CIFAR-10 (%)      |  ImageNet (%)      |  ImageNet (%)      |
| :--------- | :----------------: | :----------------: | :----------------: | :----------------: |
|            |  **milestone**     |  **cosine**        |  **milestone**     |  **cosine**        |
| SGD        |  93.94 (0.07)      |  94.53 (0.27)      |  77.47 (0.07)      |  78.22 (0.08)      |
| Adam*      |  91.41 (0.30)      |  91.56 (0.19)      |  71.72 (0.13)      |  71.19 (0.10)      |
| RAdam*     |  91.80 (0.04)      |  91.88 (0.15)      |  72.37 (0.08)      |  71.64 (0.14)      |
| Adam       |  93.74 (0.15)      |  94.24 (0.09)      |  76.86 (0.06)      |  77.54 (0.16)      |
| RAdam      |  93.88 (0.11)      |  94.38 (0.25)      |  76.91 (0.07)      |  77.68 (0.08)      |
| AdaHessian |  93.97 (0.22)      |  94.48 (0.17)      |   ---              |   ---              |
| **Apollo** |  **94.21 (0.08)**  |  **94.64 (0.09)**  |  **77.85 (0.07)**  |  **78.45 (0.06)**  |

We use [ResNet-110](https://github.com/bearpaw/pytorch-classification) for CIFAR-10 and standard ResNext-50 for ImageNet. 
Note that ResNet-110 is a modified version of ResNet-18 to adapt the small image size ```32x32``` in CIFAR-10.
ResNet-110 is much smaller than ResNet-18, with ```1.73M``` parameters (ResNet-18 has ```11.69M``` parameters).

The following table summarizes the key hyper-parameters for different optimizers. 
For the model training of image classification, please go to this [folder](https://github.com/XuezheMax/apollo/tree/master/classification).

**ResNet-110 on CIFAR-10**

|  Method    |    lr      |  weight decay  |  decoupled weight decay |  eps    | warmup updates  |  init_lr  |
| :--------- | :--------: | :------------: | :---------------------: | :-----: | :--------------: | :-------: |
|  SGD       |   0.1      |      5e-4      |         False           |   NA    |        0         |     NA    |
|  Adam*     |   0.001    |      5e-4      |         True            |  1e-8   |        0         |     NA    |
|  RAdam*    |   0.001    |      5e-4      |         True            |  1e-8   |        0         |     NA    |
|  Adam      |   0.001    |      2.5e-1    |         True            |  1e-8   |        0         |     NA    |
|  RAdam     |   0.001    |      2.5e-1    |         True            |  1e-8   |        0         |     NA    |
|  AdaHessian|   0.15     |      1e-3      |         True            |  1e-2   |       200        |    0.001  |
|  Apollo    |   1.0      |      2.5e-4    |         False           |  1e-4   |       200        |    0.01   |

**ResNext-50 on ImageNet**

|  Method    |    lr      |  weight decay  |  decoupled weight decay |  eps    |  warmup updates  |  init_lr  |
| :--------- | :--------: | :------------: | :---------------------: | :-----: | :--------------: | :-------: |
|  SGD       |   0.1      |      1e-4      |         False           |   NA    |        0         |     NA    |
|  Adam      |   0.001    |      1e-4      |         True            |  1e-8   |        0         |     NA    |
|  RAdam     |   0.001    |      1e-4      |         True            |  1e-8   |        0         |     NA    |
|  Adam-adj  |   0.001    |      1e-1      |         True            |  1e-8   |        0         |     NA    |
|  RAdam-adj |   0.001    |      1e-1      |         True            |  1e-8   |        0         |     NA    |
|  Apollo    |   1.0      |      1e-4      |         False           |  1e-4   |       200        |    0.01   |

Note that decoupled weight decay is applied to both Adam and RAdam. 
To adjust the weight decay rates of Adam and RAdam, we multiply the original rate with ```500```, which is the ratio of ```lr_apollo / lr_adam```.

### Language Modeling
<img src="./docs/images/language_model.png" width="1000"/>

| Method     |  Test PPL        |
| :--------- | :--------------: |
| Adam       |  36.68 (0.21)    |
| RAdam      |  36.20 (0.38)    |
| **Apollo** | **31.94 (0.09)** |

We use 2-layer LSTMs with ```2048``` hidden size on [One Billion Words](https://www.statmt.org/lm-benchmark/).
Some key hyper-parameters are listed in the following table. 
For the model training of language modeling, please go to this [folder](https://github.com/XuezheMax/apollo/tree/master/language_model).

**2-layer LSTM on One Billion Words**

|  Method    |    lr      |  weight decay  |  decoupled weight decay |  warmup updates  |  init_lr  |  gradient clip  |
| :--------- | :--------: | :------------: | :---------------------: | :--------------: | :-------: | :-------------: |
|  Adam      |   0.001    |      0         |         True            |        0         |     NA    |      1.0        |
|  RAdam     |   0.001    |      0         |         True            |        0         |     NA    |      1.0        |
|  Apollo    |   10.0     |      0         |         False           |       400        |    0.01   |      1.0        |

Since the weight decay rate is zero for all the optimizers, there is no difference of decoupled weight decay.

### Neural Machine Translation

| Method     |  Test BLEU       |
| :--------  |  :-------------: |
| SGD        |  26.59 (0.07)    |
| Adam       |  27.84 (0.12)    |
| RAdam      |  28.15 (0.15)    |
| **Apollo** | **28.34 (0.10)** |

We use the Transformer-base models.
Some key hyper-parameters are listed in the following table.
For the details of NMT experiments, please go to this [repo](https://github.com/XuezheMax/fairseq-apollo).

**Transformer-base on WMT-14 En-De**

|  Method    |    lr      |  weight decay  |  decoupled weight decay |    lr scheduler     |  warmup updates  |  init_lr  |  gradient clip  |
| :--------- | :--------: | :------------: | :---------------------: | :-----------------: | :--------------: | :-------: | :-------------: |
|  SGD       |   0.1      |      1e-6      |         False           |      milestone      |       1000       |    1e-4   |      1.0        |
|  Adam      |   0.0005   |      1e-4      |         True            |    inverse sqrt     |       4000       |    1e-7   |      1.0        |
|  RAdam     |   0.0005   |      1e-4      |         True            |      milestone      |        0         |     NA    |      1.0        |
|  Apollo    |   10.0     |      1e-8      |         False           |      milestone      |       1000       |    0.01   |      1.0        |

## Discussion
**1. Weight Decay:**

- The strength of `weight decay` has significant impact on both the performance of convergence speed and generalization accuracy.
Moreover, different implementations of `weight decay`, such as the decoupled version, lead to very different regularization strength with the same `weight decay rate`.
Thus, as discussed in the paper, we suggest to consider the effect of regularization strength when we analyze the performance of different optimization methods.

- In this paper, for fair comparison, we comprehensively tune the `learning rate` and the `weight decay rate` for all the optimizers on CIFAR-10. 
For ImageNet, due to the resource limits, we kept all the hyper-parameters selected from CIFAR-10 for each optimizer, and only tuned the `weight decay rate`.
One motivation is to test the consistency of hyper-parameters of these optimizers on different tasks.

- We analyzed the effect of different `weight decay rates` on different optimizers. 
As illustrated in the figure, Apollo achieves improvements over all the three baselines on convergence speed with different rates of weight decay

<img src="./docs/images/cifar10_wd.png" width="2000"/>

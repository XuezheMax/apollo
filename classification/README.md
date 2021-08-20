# Image Classification

This folder is modified based on the pytorch classification project [original repo](https://github.com/bearpaw/pytorch-classification).

For all experiments, the recommended random seeds are in {1, 101, 8191, 65537, 131071, 524287, 6700417}.

## Table of Contents

- [CIFAR-10](#cifar-10-experiments)
- [ImageNet](#imagenet-experiments)

## CIFAR-10 Experiments
### SGD
#### milestone decay
```base
python -u cifar.py --depth 110 --batch_size 128 --epochs 164 \
    --opt sgd --lr 0.1 --opt_h1 0.9 --weight_decay 5e-4 \
    --lr_decay milestone --milestone 80 120 --decay_rate 0.1 \
    --dataset cifar10 --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
#### cosine annealing
```base
python -u cifar.py --depth 110 --batch_size 128 --epochs 200 \
    --opt sgd --lr 0.1 --opt_h1 0.9 --weight_decay 5e-4 \
    --lr_decay cosine --last_lr 0.0001 --decay_rate 0.1 \
    --dataset cifar10 --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
### Adam, RAdam & AdaBelief
#### milestone decay
```base
python -u cifar.py --depth 110 --batch_size 128 --epochs 164 \
    --opt [adamw|radamw|adabelief] --lr 0.001 --opt_h1 0.9 --opt_h2 0.999 --eps 1e-8 \
    --weight_decay 2.5e-2 --weight_decay_type 'decoupled' \
    --lr_decay milestone --milestone 80 120 --decay_rate 0.1 \
    --dataset cifar10 --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
#### cosine annealing
```base
python -u cifar.py --depth 110 --batch_size 128 --epochs 200 \
    --opt [adamw|radamw|adabelief] --lr 0.001 --opt_h1 0.9 --opt_h2 0.999 --eps 1e-8 \
    --weight_decay 2.5e-2 --weight_decay_type 'decoupled' \
    --lr_decay cosine --last_lr 1e-6 --decay_rate 0.1 \
    --dataset cifar10 --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
For Adam* & RAdam*, change `--weight_decay` from `2.5e-2` to `5e-4`.
### Apollo
#### milestone decay
```base
python -u cifar.py --depth 110 --batch_size 128 --epochs 164 \
    --opt apollo --lr 0.01 --opt_h1 0.9 --eps 1e-4 \
    --weight_decay 2.5e-4 --weight_decay_type 'L2' \
    --lr_decay milestone --milestone 80 120 --decay_rate 0.1 \
    --warmup_updates 500 --init_lr 1e-5 \
    --dataset cifar10 --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
#### cosine annealing
```base
python -u cifar.py --depth 110 --batch_size 128 --epochs 200 \
    --opt apollo --lr 0.01 --opt_h1 0.9 --eps 1e-4 \
    --weight_decay 2.5e-4 --weight_decay_type 'L2' \
    --lr_decay cosine --last_lr 1e-5 --decay_rate 0.1 \
    --warmup_updates 500 --init_lr 1e-5 \
    --dataset cifar10 --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```

## ImageNet Experiments
For ImageNet, we use the ResNeXt-50 architecture, and each experiment is conducted on 8 NVIDIA Tesla V100 GPUs.
For distributed training on multi-GPUs, please refer to the pytorch distributed parallel training [tutorial](https://pytorch.org/tutorials/intermediate/dist_tuto.html).
### SGD
#### milestone decay
```base
python -u imagenet.py --nproc_per_node 8 --master_addr <ip addr> --master_port <port> \
    --arch resnext50 --batch_size 256 --epochs 120 \
    --opt sgd --lr 0.1 --opt_h1 0.9 --weight_decay 1e-4 \
    --lr_decay milestone --milestone 40 80 --decay_rate 0.1 \
    --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
#### cosine annealing
```base
python -u imagenet.py --nproc_per_node 8 --master_addr <ip addr> --master_port <port> \
    --arch resnext50 --batch_size 256 --epochs 120 \
    --opt sgd --lr 0.1 --opt_h1 0.9 --weight_decay 1e-4 \
    --lr_decay cosine --last_lr 0.0001 --decay_rate 0.1 \
    --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
### Adam, RAdam & AdaBelief
#### milestone decay
```base
python -u imagenet.py --nproc_per_node 8 --master_addr <ip addr> --master_port <port> \
    --arch resnext50 --batch_size 256 --epochs 120 \
    --opt [adamw|radamw|adabelief] --lr 0.001 --opt_h1 0.9 --opt_h2 0.999 --eps 1e-8 \
    --weight_decay 1e-1 --weight_decay_type 'decoupled' \
    --lr_decay milestone --milestone 40 80 --decay_rate 0.1 \
    --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
#### cosine annealing
```base
python -u imagenet.py --nproc_per_node 8 --master_addr <ip addr> --master_port <port> \
    --arch resnext50 --batch_size 256 --epochs 120 \
    --opt [adamw|radamw|adabelief] --lr 0.001 --opt_h1 0.9 --opt_h2 0.999 --eps 1e-8 \
    --weight_decay 1e-1 --weight_decay_type 'decoupled' \
    --lr_decay cosine --last_lr 1e-6 --decay_rate 0.1 \
    --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
For Adam* & RAdam*, change `--weight_decay` from `1e-1` to `1e-4`.
### Apollo
#### milestone decay
```base
python -u imagenet.py --nproc_per_node 8 --master_addr <ip addr> --master_port <port> \
    --arch resnext50 --batch_size 256 --epochs 120 \
    --opt apollo --lr 0.01 --opt_h1 0.9 --eps 1e-4 \
    --weight_decay 1e-4 --weight_decay_type 'L2' \
    --lr_decay milestone --milestone 40 80 --decay_rate 0.1 \
    --warmup_updates 500 --init_lr 1e-5 \
    --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```
#### cosine annealing
```base
python -u imagenet.py --nproc_per_node 8 --master_addr <ip addr> --master_port <port> \
    --arch resnext50 --batch_size 256 --epochs 120 \
    --opt apollo --lr 0.01 --opt_h1 0.9 --eps 1e-4 \
    --weight_decay 1e-4 --weight_decay_type 'L2' \
    --lr_decay cosine --last_lr 1e-5 --decay_rate 0.1 \
    --warmup_updates 500 --init_lr 1e-5 \
    --data_path <data path> --model_path <model path> \
    --run <run_id> --seed <random seed> 
```

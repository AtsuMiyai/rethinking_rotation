# Rethinking Rotation in Self-Supervised Contrastive Learning: Adaptive Positive or Negative Data Augmentation
This repository provides the training and evaluation codes for SimCLR PNDA.

## 1. Requirements
### Environments
Currently, requires following packages
- python 3.7
- torch 1.4
- torchvision 0.5
- CUDA 10.1+
- tensorboard 2.0
- tensorboardX
- [torchlars](https://github.com/kakaobrain/torchlars) == 0.1.2 
- [pytorch-gradual-warmup-lr](https://github.com/ildoonet/pytorch-gradual-warmup-lr) packages 

We have done these codes for SimCLR with a single Nvidia V100 GPU.

## 2. Training
Please refer to `run_train.sh`. 

## 3. Inference
Please refer to `run_inference.sh`. 
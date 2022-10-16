# Rethinking Rotation in Self-Supervised Contrastive Learning: Adaptive Positive or Negative Data Augmentation

This repository provides the training and evaluation codes for MoCo v2 PNDA.

## 1. Requirements

### Environments

- python 3.7.4
- numpy 1.17.2
- torch 1.4.0
- torchvision 0.5.0
- tensorboard
- cudatoolkit 10.1

We have done these codes for MoCo v2 witpip h 4 Nvidia V100 GPUs.   


## 2. Running scripts (Training + Inference)

Please refer to `run.sh`.

The scores in the paper are the highest scores from those with several learning rates {0.5, 1.5, 2.5, 5, 15, 25, and 35} during linear evaluation.

## Note

This code is adapted from [imix](https://github.com/kibok90/imix).

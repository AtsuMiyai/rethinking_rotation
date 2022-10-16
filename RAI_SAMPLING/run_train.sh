# CIFAR100
CUDA_VISIBLE_DEVICES=0 python train.py '../data/' --dataset cifar100 --model resnet18 --suffix 1time
# Tiny ImageNet
CUDA_VISIBLE_DEVICES=0 python train.py  '../data/tiny-imagenet-200/' --dataset tiny_imagenet --model resnet18_imagenet --suffix 1time

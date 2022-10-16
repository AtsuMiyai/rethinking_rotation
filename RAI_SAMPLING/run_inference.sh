# CIFAR100
CUDA_VISIBLE_DEVICES=0 python inference.py '../data/' --dataset cifar100 --model resnet18 --load_path <model_path>
# Tiny ImageNet
CUDA_VISIBLE_DEVICES=0 python inference.py '../data/tiny-imagenet-200/' --dataset tiny_imagenet --model resnet18_imagenet --load_path  <model_path>
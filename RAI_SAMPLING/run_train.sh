# CIFAR100
CUDA_VISIBLE_DEVICES=2 python train.py '~/data/' --dataset cifar100 --model resnet18 --batch_size 64 --optimizer 'adam' --lr_init 1e-3 --weight_decay 1e-6 --epochs_beta_1 10  --epochs_beta_2 200 --lambda_ 0.20 --beta_1_overacc 90 --suffix 1time

# Tiny ImageNet
CUDA_VISIBLE_DEVICES=2 python train.py '~/data/tiny-imagenet-200/' --dataset tiny_imagenet --model resnet18_imagenet --batch_size 64 --optimizer 'sgd' --lr_init 0.1 --weight_decay 1e-6 --epochs_beta_1 10  --epochs_beta_2 150 --lambda_ 0.10 --beta_1_overacc 78 --suffix 1time

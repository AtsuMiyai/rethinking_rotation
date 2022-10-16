# please choose "mode" from {SimCLR, Simclr_PDA, SimCLR_NDA, SimCLR_PNDA}

# CIFAR-100, ResNet-18*
CUDA_VISIBLE_DEVICES=0 python main_lincls.py '../data/' --dataset cifar100 --model resnet18 --batch_size 128 --suffix simclr_eval --load_path <MODEL_PATH> --suffix <mode>
 
# CIFAR-100, ResNet-50*
CUDA_VISIBLE_DEVICES=0 python main_lincls.py '../data/' --dataset cifar100 --model resnet50 --batch_size 128 --suffix simclr_eval --load_path <MODEL_PATH> --suffix <mode>

# Tiny ImageNet, ResNet-18
CUDA_VISIBLE_DEVICES=0 python main_lincls.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet18_imagenet --batch_size 128 --suffix simclr_eval --load_path <MODEL_PATH> --suffix <mode>

# Tiny ImageNet, ResNet-50
CUDA_VISIBLE_DEVICES=0 python main_lincls.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet50_imagenet --batch_size 128 --suffix simclr_eval --load_path <MODEL_PATH> --suffix <mode>
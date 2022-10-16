#############
# table 3 (a)#
#############

# SimCLR
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/' --dataset cifar100 --model resnet18 --mode simclr --batch_size 128 --lr_init 0.20
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/' --dataset cifar100 --model resnet50 --mode simclr --batch_size 128 --lr_init 0.20

# SimCLR+PDA 
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/' --dataset cifar100 --model resnet18 --mode simclr_pda --batch_size 128 --lr_init 0.20
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/' --dataset cifar100 --model resnet50 --mode simclr_pda --batch_size 128 --lr_init 0.20

# SimCLR+NDA 
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/' --dataset cifar100 --model resnet18 --mode simclr_nda --batch_size 128 --lr_init 0.20
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/'--dataset cifar100 --model resnet50 --mode simclr_nda --batch_size 128 --lr_init 0.20

# SimCLR+PNDA 
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/' --dataset cifar100 --model resnet18 --mode simclr_pnda --batch_size 128 --nega_posi_flag ../RAI_results/cifar100_nega_posi_flag.pkl --lr_init 0.20
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/' --dataset cifar100 --model resnet50 --mode simclr_pnda --batch_size 128 --nega_posi_flag ../RAI_results/cifar100_nega_posi_flag.pkl --lr_init 0.20

#############
# table 3 (b)#
#############
# SimCLR
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet18_imagenet --mode simclr --batch_size 256 --lr_init 0.20  --epochs 200
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet50_imagenet --mode simclr --batch_size 256 --lr_init 0.20  --epochs 200

# SimCLR+PDA 
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet18_imagenet --mode simclr_pda --batch_size 256 --lr_init 0.20  --epochs 200
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet50_imagenet --mode simclr_pda --batch_size 256 --lr_init 0.20  --epochs 200

# SimCLR+NDA 
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet18_imagenet --mode simclr_nda --batch_size 256 --lr_init 0.20  --epochs 200
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet50_imagenet --mode simclr_nda --batch_size 256 --lr_init 0.20  --epochs 200

# SimCLR+PNDA 
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet18_imagenet --mode simclr_pnda --batch_size 256 --nega_posi_flag ../RAI_results/tin_nega_posi_flag.pkl --lr_init 0.20  --epochs 200
CUDA_VISIBLE_DEVICES=0 python main_simclr_pnda.py '../data/tiny-imagenet-200' --dataset tiny_imagenet --model resnet50_imagenet --mode simclr_pnda --batch_size 256 --nega_posi_flag ../RAI_results/tin_nega_posi_flag.pkl --lr_init 0.20  --epochs 200


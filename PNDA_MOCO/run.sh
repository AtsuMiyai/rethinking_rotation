# Training + Inference code
# 4 gpus are used

#############
# table 2 (a)#
#############

# MoCo v2
python main_moco_pnda.py --mode moco '../data' --dataset cifar100 -a resnet18 --cos --warm --lincls --tb --resume true --method moco --proj mlpbn1 --temp 0.2 --epochs 300 --multiprocessing-distributed --dist-url 'tcp://localhost:10001' --lr 0.250 --trial 1time >> log_cifar100_resnet18*_moco.txt
python main_moco_pnda.py --mode moco '../data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method moco --proj mlpbn1 --temp 0.2 --epochs 300 --multiprocessing-distributed --dist-url 'tcp://localhost:10002' --lr 0.250 --trial 1time >> log_cifar100_resnet50*_moco.txt

# MoCo v2 + PDA 
python main_moco_pnda.py --mode moco_pda '../data' --dataset cifar100 -a resnet18 --cos --warm --lincls --tb --resume true --method moco --proj mlpbn1 --temp 0.2 --epochs 300  --multiprocessing-distributed --dist-url 'tcp://localhost:10003' --lr 0.250 --trial 1time >> log_cifar100_resnet18*_moco_pda.txt
python main_moco_pnda.py --mode moco_pda '../data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method moco --proj mlpbn1 --temp 0.2 --epochs 300  --multiprocessing-distributed --dist-url 'tcp://localhost:10004' --lr 0.250 --trial 1time >> log_cifar100_resnet50*_moco_pda.txt 

# MoCo v2 + NDA 
python main_moco_pnda.py --mode moco_nda '../data' --dataset cifar100 -a resnet18 --cos --warm --lincls --tb --resume true --method moco --proj mlpbn1 --temp 0.2 --epochs 300  --multiprocessing-distributed --dist-url 'tcp://localhost:10005' --lr 0.250 --trial 1time >> log_cifar100_resnet18*_moco_nda.txt
python main_moco_pnda.py --mode moco_nda '../data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method moco --proj mlpbn1 --temp 0.2 --epochs 300  --multiprocessing-distributed --dist-url 'tcp://localhost:10006' --lr 0.250 --trial 1time >> log_cifar100_resnet50*_moco_nda.txt 

# MoCo v2 + PNDA 
python main_moco_pnda.py --mode moco_pnda '../data' --dataset cifar100 -a resnet18 --cos --warm --lincls --tb --resume true --method moco --proj mlpbn1 --temp 0.2 --epochs 300  --multiprocessing-distributed --dist-url 'tcp://localhost:10007' --lr 0.250 --nega_posi_flag ../RAI_results/cifar100_nega_posi_flag.pkl --trial 1time >> log_cifar100_resnet18*_moco_pnda.txt 
python main_moco_pnda.py --mode moco_pnda '../data' --dataset cifar100 -a resnet50 --cos --warm --lincls --tb --resume true --method moco --proj mlpbn1 --temp 0.2 --epochs 300  --multiprocessing-distributed --dist-url 'tcp://localhost:10008' --lr 0.250 --nega_posi_flag ../RAI_results/cifar100_nega_posi_flag.pkl --trial 1time >> log_cifar100_resnet50*_moco_pnda.txt

#############
# table 2 (b)#
#############

# MoCo v2
python main_moco_pnda.py --mode moco '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet18 --small 0 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --trial 1time >> log_tin_resnet18_moco.txt
python main_moco_pnda.py --mode moco '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet50 --small 0 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --trial 1time >> log_tin_resnet50_moco.txt
python main_moco_pnda.py --mode moco '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet18 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256  --trial 1time >> log_tin_resnet18*_moco.txt

# MoCo v2 + PDA 
python main_moco_pnda.py  --mode moco_pda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet18 --small 0 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --trial 1time >> log_tin_resnet18_moco_pda.txt
python main_moco_pnda.py  --mode moco_pda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet50 --small 0 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --trial 1time >> log_tin_resnet50_moco_pda.txt 
python main_moco_pnda.py  --mode moco_pda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet18 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --trial 1time >> log_tin_resnet18*_moco_pda.txt

# MoCo v2 + NDA 
python main_moco_pnda.py  --mode moco_nda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet18 --small 0 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --trial 1time >> log_tin_resnet18_moco_nda.txt 
python main_moco_pnda.py  --mode moco_nda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet50 --small 0 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --trial 1time >> log_tin_resnet50_moco_nda.txt  
python main_moco_pnda.py  --mode moco_nda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet18  --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --trial 1time >> log_tin_resnet18*_moco_nda.txt 

# MoCo v2 + PNDA 
python main_moco_pnda.py  --mode moco_pnda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet18 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --small 0  --nega_posi_flag ../RAI_results/tin_nega_posi_flag.pkl --trial 1time >> log_tin_resnet18_moco_pnda.txt  
python main_moco_pnda.py  --mode moco_pnda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet50 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --small 0  --nega_posi_flag ../RAI_results/tin_nega_posi_flag.pkl --trial 1time >> log_tin_resnet50_moco_pnda.txt   
python main_moco_pnda.py  --mode moco_pnda '../data/tiny-imagenet-200' --dataset tiny_imagenet -a resnet18 --cos --warm --lincls --tb --resume true --method moco  --proj mlpbn1 --temp 0.2 --epochs 200 --multiprocessing-distributed --dist-url 'tcp://localhost:10010' --lr 0.125 --batch-size 256 --small 1  --nega_posi_flag ../RAI_results/tin_nega_posi_flag.pkl --trial 1time >> log_tin_resnet18*_moco_pnda.txt   

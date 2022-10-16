from argparse import ArgumentParser
import torch
from torch.utils.data import DataLoader
import models.classifier as C
from datasets import get_dataset
from evals import calculate_score
import math
import pickle
import time

parser = ArgumentParser(description='Pytorch implementation of SimCLR PNDA')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', help='Dataset',
                    choices=['cifar100', 'tiny_imagenet'], type=str)
parser.add_argument('--model', help='Model',
                    choices=['resnet18', 'resnet18_imagenet'], type=str)
parser.add_argument('--load_path', help='Path to the pretrained model',
                        default=None, type=str)
parser.add_argument('--batch_size', help='batch size for inference',
                        default=64, type=int)

P = parser.parse_args()

### Set torch device ###
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()
P.S_size = 4  # {0, 90, 180, 270}

P.pho = math.log(4)/2
P.m = 0.2 

model = C.get_classifier(
    P.model, n_classes=P.S_size).to(device)


checkpoint = torch.load(P.load_path)
model.load_state_dict(checkpoint, strict=False)

model = model.to(device)
_, infer_set, image_size, _ = get_dataset(P, dataset=P.dataset)
P.image_size = image_size
P.infer_set = len(infer_set)

kwargs = {'pin_memory': False, 'num_workers': 4}

infer_loader = DataLoader(infer_set, shuffle=False,
                            batch_size=P.batch_size, **kwargs)

score_list = calculate_score(P, model, infer_loader)

threshold = P.pho + P.m

nega_posi_flag = {}
for i, score in enumerate(score_list):
    nega_posi_flag[i] = 1 if score > threshold else 0

print(f'#RAI {sum(nega_posi_flag.values())}')

file_name = f'{P.dataset}_nega_posi_flag_{time.time()}.pkl'
with open(file_name, "wb") as tf:
    pickle.dump(nega_posi_flag, tf)
from utils.utils import Logger
from utils.utils import save_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
from training.scheduler import GradualWarmupScheduler
import models.classifier as C
from datasets import get_dataset
from training.unsup import setup
from argparse import ArgumentParser
from utils.utils import load_checkpoint


##### Command-line argument parser for training. #####

parser = ArgumentParser(description='Pytorch implementation of SimCLR PNDA')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', help='Dataset',
                    choices=['cifar100', 'tiny_imagenet'], type=str)
parser.add_argument('--model', help='Model',
                    choices=['resnet18', 'resnet50', 'resnet18_imagenet', 'resnet50_imagenet'], type=str)
parser.add_argument('--mode', help='Training mode',
                    choices=['simclr', 'simclr_pda', 'simclr_nda', 'simclr_pnda'], type=str)
parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                    default=128, type=int)
parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
parser.add_argument('--suffix', help='Suffix for the log dir',
                    default=None, type=str)
parser.add_argument('--save_step', help='Epoch steps to save models',
                    default=10, type=int)
parser.add_argument("--resize_factor", help='resize scale is sampled from [resize_factor, 1.0]',
                    default=0.08, type=float)
parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                    action='store_true')

##### Training Configurations #####
parser.add_argument('--epochs', help='Epochs',
                    default=300, type=int)
parser.add_argument('--optimizer', help='Optimizer',
                    choices=['sgd', 'lars'],
                    default='lars', type=str)
parser.add_argument('--lr_scheduler', help='Learning rate scheduler',
                    choices=['step_decay', 'cosine'],
                    default='cosine', type=str)
parser.add_argument('--warmup', help='Warm-up epochs',
                    default=10, type=int)
parser.add_argument('--lr_init', help='Initial learning rate',
                    default=1e-1, type=float)
parser.add_argument('--weight_decay', help='Weight decay',
                    default=1e-6, type=float)
parser.add_argument('--batch_size', help='Batch size',
                    default=128, type=int)
parser.add_argument('--nega_posi_flag', help='flag for posi_nega',
                    default='', type=str)
##### Objective Configurations #####
parser.add_argument('--sim_lambda', help='Weight for SimCLR loss',
                    default=1.0, type=float)
parser.add_argument('--temperature', help='Temperature for similarity',
                    default=0.5, type=float)

P = parser.parse_args()

### Set torch device ###
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()

P.multi_gpu = False

### Initialize dataset ###
train_set, _, image_size, n_classes = get_dataset(P, dataset=P.dataset)
P.image_size = image_size
P.n_classes = n_classes  # not use for pre-training stage

kwargs = {'pin_memory': False, 'num_workers': 4}


train_loader = DataLoader(train_set, shuffle=True,
                              batch_size=P.batch_size, **kwargs)

simclr_aug = C.get_simclr_augmentation(P, image_size=P.image_size).to(device)
criterion = nn.CrossEntropyLoss().to(device)
model = C.get_classifier(
    P.model, n_classes=P.n_classes).to(device)  

if P.optimizer == 'sgd':
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    lr_decay_gamma = 0.1
elif P.optimizer == 'lars':
    from torchlars import LARS
    base_optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    optimizer = LARS(base_optimizer, eps=1e-8, trust_coef=0.001)
    lr_decay_gamma = 0.1
else:
    raise NotImplementedError()


if P.lr_scheduler == 'cosine':
    scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)
elif P.lr_scheduler == 'step_decay':
    milestones = [int(0.5 * P.epochs), int(0.75 * P.epochs)]
    scheduler = lr_scheduler.MultiStepLR(optimizer, gamma=lr_decay_gamma, milestones=milestones)
else:
    raise NotImplementedError()

scheduler_warmup = GradualWarmupScheduler(
    optimizer, multiplier=10.0, total_epoch=P.warmup, after_scheduler=scheduler)

if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=False)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch'] + 1
else:
    resume = False
    start_epoch = 1

train, fname = setup(P.mode, P)

logger = Logger(fname, ask=not resume)
logger.log(P)
logger.log(model)


for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    kwargs = {}
    kwargs['simclr_aug'] = simclr_aug

    train(P, epoch, model, criterion, optimizer,
          scheduler_warmup, train_loader, logger=logger, **kwargs)

    model.eval()

    if epoch % P.save_step == 0 or epoch == P.epochs:

        save_states = model.state_dict()
        save_checkpoint(epoch, save_states,
                        optimizer.state_dict(), logger.logdir)
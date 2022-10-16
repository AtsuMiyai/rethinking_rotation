from utils.utils import Logger
from utils.utils import save_checkpoint
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as lr_scheduler
from torch.utils.data import DataLoader
import models.classifier as C
from datasets import get_dataset
from argparse import ArgumentParser
from training.predictor_train import train
from evals import test_classifier
from utils.utils import load_checkpoint

##### Command-line argument parser for training. #####

parser = ArgumentParser(description='Pytorch implementation of SimCLR PNDA')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', help='Dataset',
                    choices=['cifar100', 'tiny_imagenet'], type=str)
parser.add_argument('--model', help='Model',
                    choices=['resnet18', 'resnet18_imagenet'], type=str)
parser.add_argument('--resume_path', help='Path to the resume checkpoint',
                        default=None, type=str)
parser.add_argument('--suffix', help='Suffix for the log dir',
                    default=None, type=str)
parser.add_argument('--save_step', help='Epoch steps to save models',
                    default=10, type=int)

P = parser.parse_args()
if P.suffix is None:
    fname = f'{P.dataset}_{P.model}'
else:
    fname = f'{P.dataset}_{P.model}' + f'_{P.suffix}'

### Set torch device ###
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()
P.S_size = 4  # {0, 90, 180, 270}

model = C.get_classifier(
    P.model, n_classes=P.S_size).to(device)

### Set hyperparameters ###
if P.dataset == 'cifar100':
    assert P.model == 'resnet18'
    P.batch_size = 64
    P.optimizer = 'adam'
    P.lr_init = 1e-3
    P.weight_decay = 1e-6
    optimizer = optim.Adam(model.parameters(), lr=P.lr_init, betas=(.9, .999), weight_decay=P.weight_decay) 
    P.beta_1 = 10  # epochs for step1
    P.beta_2 = 200  # epochs for step2
    P.epochs = 210  # epochs for step1+step2
    P.lambda_ = 0.20
    P.margin = 0.20
    P.beta_1 = 10
    P.beta_1_overacc = 90
elif P.dataset == 'tiny_imagenet':
    assert P.model == 'resnet18_imagenet'
    P.batch_size = 64
    P.optimizer = 'sgd'
    P.lr_init = 1e-1
    P.weight_decay = 1e-6
    optimizer = optim.SGD(model.parameters(), lr=P.lr_init, momentum=0.9, weight_decay=P.weight_decay)
    P.beta_1 = 10  # epochs for step1
    P.beta_2 = 150  # epochs for step2
    P.epochs = 160  # epochs for step1+step2
    P.lambda_ = 0.10
    P.margin = 0.20
    P.beta_1_overacc = 78
else:
    raise NotImplementedError()

### Initialize dataset ###
train_set, infer_set, image_size, _ = get_dataset(P, dataset=P.dataset)
P.image_size = image_size
P.infer_size = len(infer_set)

kwargs = {'pin_memory': False, 'num_workers': 4}

train_loader = DataLoader(train_set, shuffle=True,
                            batch_size=P.batch_size, **kwargs)
infer_loader = DataLoader(infer_set, shuffle=False,
                            batch_size=P.batch_size, **kwargs)

criterion = nn.CrossEntropyLoss().to(device)

scheduler = lr_scheduler.CosineAnnealingLR(optimizer, P.epochs)

if P.resume_path is not None:
    resume = True
    model_state, optim_state, config = load_checkpoint(P.resume_path, mode='last')
    model.load_state_dict(model_state, strict=False)
    optimizer.load_state_dict(optim_state)
    start_epoch = config['epoch'] + 1
else:
    resume = False
    start_epoch = 1


logger = Logger(fname, ask=not resume)
logger.log(P)
logger.log(model)


for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()
    model.eval()
    train(P, epoch, model, criterion, optimizer,
          scheduler, train_loader, logger=logger)

    # overfit_check
    if epoch == P.beta_1:
        rot_acc = test_classifier(P, model=model,  test_loader=infer_loader)
        logger.log('[rot_acc %.3f]' % (rot_acc))
        assert rot_acc < P.beta_1_overacc, "It is likely that this model are overfitting, please try again."

    if epoch % P.save_step == 0:
        save_states = model.state_dict()
        save_checkpoint(epoch, save_states,
                        optimizer.state_dict(), logger.logdir)

final_rot_acc = test_classifier(P, model=model,  test_loader=infer_loader)
logger.log('[final_rot_acc %.3f]' % (final_rot_acc))
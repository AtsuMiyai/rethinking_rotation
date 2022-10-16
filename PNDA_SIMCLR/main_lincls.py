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
from training.sup import setup
from argparse import ArgumentParser
from evals import test_classifier


##### Command-line argument parser for training. #####

parser = ArgumentParser(description='Pytorch implementation of SimCLR PNDA')
parser.add_argument('data', metavar='DIR',
                    help='path to dataset')
parser.add_argument('--dataset', help='Dataset',
                    choices=['cifar100', 'tiny_imagenet'], type=str)
parser.add_argument('--model', help='Model',
                    choices=['resnet18', 'resnet50', 'resnet18_imagenet', 'resnet50_imagenet'], type=str)
parser.add_argument('--simclr_dim', help='Dimension of simclr layer',
                    default=128, type=int)
parser.add_argument("--no_strict", help='Do not strictly load state_dicts',
                    action='store_true')
parser.add_argument('--suffix', help='Suffix for the log dir',
                    default=None, type=str)
parser.add_argument('--error_step', help='Epoch steps to compute errors',
                        default=1, type=int)
parser.add_argument('--save_step', help='Epoch steps to save models',
                    default=2, type=int)
parser.add_argument("--resize_fix", help='resize scale is fixed to resize_factor (not (resize_factor, 1.0])',
                    action='store_true')
parser.add_argument('--load_path', help='Path to the loading checkpoint',
                    default=None, type=str)
parser.add_argument('--mode', default='sup_linear', type=str)
##### Training Configurations #####
parser.add_argument('--epochs', help='Epochs',
                    default=90, type=int)
parser.add_argument('--weight_decay', help='Weight decay',
                    default=1e-6, type=float)
parser.add_argument('--batch_size', help='Batch size',
                    default=128, type=int)
parser.add_argument('--test_batch_size', help='Batch size for test loader',
                        default=100, type=int)

P = parser.parse_args()

### Set torch device ###
if torch.cuda.is_available():
    torch.cuda.set_device(0)
device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")

P.n_gpus = torch.cuda.device_count()


P.multi_gpu = False

### Initialize dataset ###
train_set, test_set, image_size, n_classes = get_dataset(P, dataset=P.dataset)
P.image_size = image_size
P.n_classes = n_classes

kwargs = {'pin_memory': False, 'num_workers': 4}

train_loader = DataLoader(train_set, shuffle=True,
                            batch_size=P.batch_size, **kwargs)
test_loader = DataLoader(test_set, shuffle=False,
                            batch_size=P.test_batch_size, **kwargs)


criterion = nn.CrossEntropyLoss().to(device)
model = C.get_classifier(
    P.model, n_classes=P.n_classes).to(device)  # resnetを定義

resume = False
start_epoch = 1
best = 100.0
error = 100.0

assert P.load_path is not None
checkpoint = torch.load(P.load_path)
model.load_state_dict(checkpoint, strict=False)

train, fname = setup(P.mode, P)

logger = Logger(fname, ask=not resume)
logger.log(P)
logger.log(model)

linear = model.linear
 

for epoch in range(start_epoch, P.epochs + 1):
    logger.log_dirname(f"Epoch {epoch}")
    model.train()

    kwargs = {}
    kwargs['linear'] = linear

    train(P, epoch, model, criterion, train_loader, logger=logger, **kwargs)

    model.eval()

    if epoch % P.error_step == 0 or epoch == P.epochs:
        error = test_classifier(P, model, test_loader, epoch, logger=logger)

        is_best = (best > error)
        if is_best:
            best = error

        logger.scalar_summary('eval/best_error', best, epoch)
        logger.log('[Epoch %3d] [Test %5.2f] [Best %5.2f]' % (epoch, error, best))

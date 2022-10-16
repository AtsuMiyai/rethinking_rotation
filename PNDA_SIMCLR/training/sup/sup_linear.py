import time

import torch.optim
import torch.optim.lr_scheduler as lr_scheduler

import models.transform_layers as TL
from utils.utils import AverageMeter, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion,loader, logger=None, linear=None):

    if epoch == 1:
        # define optimizer and save in P (argument)
        milestones = [int(0.6 * P.epochs), int(0.75 * P.epochs), int(0.9 * P.epochs)]

        linear_optim = torch.optim.SGD(linear.parameters(),
                                       lr=1e-1, weight_decay=P.weight_decay)
        P.linear_optim = linear_optim
        P.linear_scheduler = lr_scheduler.MultiStepLR(P.linear_optim, gamma=0.1, milestones=milestones)

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()

    check = time.time()
    for n, (images, _, labels, _) in enumerate(loader):
        model.eval()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()
  
        batch_size = images.size(0)
        images = images.to(device)

        labels = labels.to(device)

        _, outputs_aux = model(images, penultimate=True)
        penultimate = outputs_aux['penultimate'].detach()

        outputs = linear(penultimate)

        loss_ce = criterion(outputs, labels)

        ### CE loss ###
        P.linear_optim.zero_grad()
        loss_ce.backward()
        P.linear_optim.step()

        ### optimizer learning rate ###
        lr = P.linear_optim.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        ### Log losses ###
        losses['cls'].update(loss_ce.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossC %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['cls'].value, ))
        check = time.time()

    P.linear_scheduler.step()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossC %f]' %
         (batch_time.average, data_time.average,
          losses['cls'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_cls', losses['cls'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)

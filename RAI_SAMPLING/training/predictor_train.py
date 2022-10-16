import time
import torch
from torch import nn
import torch.optim
import models.transform_layers as TL
import math
from utils.utils import AverageMeter

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)
softmax = nn.Softmax(dim=1).to(device)
margin_value = 0.20


def entropy_margin(p, value=(math.log(4))/2, margin=margin_value, weight=None):
    p = softmax(p)
    return -torch.mean(hinge(torch.abs(-torch.sum(p * torch.log(p+1e-10), -1)
                                       - value), margin))

def hinge(input, margin=margin_value):
    return torch.clamp(input, min=margin)


def make_mask(p, value=(math.log(4))/2, margin=margin_value):
    p = softmax(p)
    entropy_per_one = -torch.sum(p * torch.log(p+1e-10), -1)
    mask_tensor = torch.ones_like(entropy_per_one)
    mask_tensor = mask_tensor * (entropy_per_one < (value-margin))
    return mask_tensor


def mask_cross_entropy(p, labels, mask_tensor):
    p = softmax(p)
    one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=4)
    cross_entropy_per_one = torch.sum(one_hot_labels * torch.log(p+1e-10), -1)
    cross_entropy_per_one = cross_entropy_per_one * mask_tensor
    return -torch.mean(cross_entropy_per_one)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None):

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['crs'] = AverageMeter()
    losses['es'] = AverageMeter()
    # losses['ratio'] = AverageMeter()

    check = time.time()
    for n, (images, labels) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images.size(0)
        images = images.to(device)
        images1, _ = hflip(images.repeat(2, 1, 1, 1)).chunk(2)
     
        labels = labels.to(device)
        images1 = torch.cat([torch.rot90(images1, k, (2, 3)) for k in
                                 range(P.S_size)])  # B -> 4B
        rot_labels = torch.cat([torch.ones_like(labels) * k for k in
                                 range(P.S_size)], 0)  # B -> 4B

        outputs = model(images1)
    
        if epoch <= P.beta_1:
            loss_crs = criterion(outputs, rot_labels)
            loss_en = torch.tensor(0)  # not use
            loss = loss_crs
            # ratio = make_mask_inv(outputs)
        else:
            loss_en = entropy_margin(outputs)
            mask_tensor = make_mask(outputs)
            # ratio = make_mask_inv(outputs)
            loss_crs = mask_cross_entropy(outputs, rot_labels, mask_tensor)
            loss = loss_crs + P.lambda_/P.beta_2 * (epoch-P.beta_1) * loss_en

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)

        losses['crs'].update(loss_crs.item(), batch_size)
        losses['es'].update(loss_en.item(), batch_size)
        # losses['ratio'].update(ratio, batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossCRS %f] [LossES %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['crs'].value, losses['es'].value))

    log_('[DONE] [Time %.3f] [Data %.3f] [LossCRS %f] [LossES %f]'
         % (batch_time.average, data_time.average,
            losses['crs'].average, losses['es'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_crs', losses['crs'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
        logger.scalar_summary('train/loss_en', losses['es'].average, epoch)


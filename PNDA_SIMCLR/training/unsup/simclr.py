import time
import torch.optim
import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, NT_xent
from utils.utils import AverageMeter, normalize

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['cls'] = AverageMeter()
    losses['sim'] = AverageMeter()

    check = time.time()
    for n, (images, _, labels, _ ) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images.size(0)
        images = images.to(device)
        images_pair = hflip(images.repeat(2, 1, 1, 1))  # 2B with hflip

        labels = labels.to(device)

        images_pair = simclr_aug(images_pair)  # transform

        _, outputs_aux = model(images_pair, simclr=True, penultimate=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        loss = NT_xent(sim_matrix, temperature=0.5) * P.sim_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)   

        ### Log losses ###
        losses['sim'].update(loss.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossSim %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr,
                  losses['sim'].value))

        check = time.time()

    log_('[DONE] [Time %.3f] [Data %.3f] [LossSim %f]' %
         (batch_time.average, data_time.average, losses['sim'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
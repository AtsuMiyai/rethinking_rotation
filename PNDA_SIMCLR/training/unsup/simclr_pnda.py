import time
import torch.optim
import models.transform_layers as TL
from training.contrastive_loss import get_similarity_matrix, Supervised_NT_xent_update_anchor
from utils.utils import AverageMeter, normalize
import random

device = torch.device(f"cuda" if torch.cuda.is_available() else "cpu")
hflip = TL.HorizontalFlipLayer().to(device)


def train(P, epoch, model, criterion, optimizer, scheduler, loader, logger=None,
          simclr_aug=None, linear_optim=None):

    assert simclr_aug is not None
    assert P.sim_lambda == 1.0  # to avoid mistake
    
    num_data = 50000 if P.dataset == 'cifar100' else 100000

    if logger is None:
        log_ = print
    else:
        log_ = logger.log

    batch_time = AverageMeter()
    data_time = AverageMeter()

    losses = dict()
    losses['sim'] = AverageMeter()

    check = time.time()
    for n, (images, nega_posi_flag, labels, instance_label) in enumerate(loader):
        model.train()
        count = n * P.n_gpus  # number of trained samples

        data_time.update(time.time() - check)
        check = time.time()

        batch_size = images.size(0)
        images = images.to(device)
        images1, images2 = hflip(images.repeat(2, 1, 1, 1)).chunk(2) 

        instance_label = instance_label.to(device)
        nega_posi_flag = nega_posi_flag.to(device)
  
        ### Create rotated images. ### 
        P.theta_num = 2
        theta_1, theta_2 = random.sample([1, 2, 3], P.theta_num)
        rot_image1 = torch.rot90(images1, theta_1, (2, 3)).detach()
        rot_image2 = torch.rot90(images2, theta_2, (2, 3)).detach()

        ### Create instance-level labels for rotated images. ### 
        '''
        For RAI, the rotated images have the same label as the original image.
        For non-RAI, the rotated image have the different label from the original image.
        '''
        rot_imagess_instance_label = instance_label.clone()
        rot_imagess_instance_label[nega_posi_flag == 0] += num_data
          
        images1 = torch.cat([images1, rot_image1])
        images2 = torch.cat([images2, rot_image2])
        instance_label = torch.cat([instance_label, rot_imagess_instance_label])

        images_pair = torch.cat([images1, images2], dim=0) 
        images_pair = simclr_aug(images_pair)  # transform

        _, outputs_aux = model(images_pair, simclr=True, penultimate=True)

        simclr = normalize(outputs_aux['simclr'])  # normalize
        sim_matrix = get_similarity_matrix(simclr, multi_gpu=P.multi_gpu)
        loss = Supervised_NT_xent_update_anchor(sim_matrix, instance_label,  temperature=0.5, multi_gpu=P.multi_gpu) * P.sim_lambda

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        scheduler.step(epoch - 1 + n / len(loader))
        lr = optimizer.param_groups[0]['lr']

        batch_time.update(time.time() - check)
        losses['sim'].update(loss.item(), batch_size)

        if count % 50 == 0:
            log_('[Epoch %3d; %3d] [Time %.3f] [Data %.3f] [LR %.5f]\n'
                 '[LossSim %f]' %
                 (epoch, count, batch_time.value, data_time.value, lr, losses['sim'].value))

    log_('[DONE] [Time %.3f] [Data %.3f] [LossSim %f]' %
         (batch_time.average, data_time.average,
          losses['sim'].average))

    if logger is not None:
        logger.scalar_summary('train/loss_sim', losses['sim'].average, epoch)
        logger.scalar_summary('train/batch_time', batch_time.average, epoch)
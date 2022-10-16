import torch
import torch.nn as nn


class Builder(nn.Module):

    def __init__(self, base_encoder, dim=128, qlen=4096, emam=0.999, temp=0.2, proj='lin', pred='none', method='moco', shuffle_bn=False, head_mul=1, sym=False, in_channels=3, small=False, distributed=False, kaiming_init=True):
        super(Builder, self).__init__()

        self.qlen = qlen
        self.emam = emam
        self.temp = temp
        self.method = method
        self.shuffle_bn = shuffle_bn
        self.sym = sym
        self.distributed = distributed

        # encoder
        self.encoder_q = base_encoder(num_classes=dim*head_mul, in_channels=in_channels, small=small, kaiming_init=kaiming_init)
        self.encoder_k = base_encoder(num_classes=dim*head_mul, in_channels=in_channels, small=small, kaiming_init=kaiming_init)

        # projection head
        dim_out, dim_in = self.encoder_q.fc.weight.shape
        dim_mlp = dim_in * head_mul
        if proj == 'mlpbn':
            print('MLP projection layer with BN')
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out), BatchNorm1d(dim_out))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out), BatchNorm1d(dim_out))
        elif proj == 'mlpbn1':
            print('MLP projection layer with BN in the middle')
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        elif proj == 'mlp':
            print('MLP projection layer without BN')
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
            self.encoder_k.fc = nn.Sequential(nn.Linear(dim_in, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        elif proj == 'linbn':
            print('Linear projection layer with BN')
            self.encoder_q.fc = nn.Sequential(nn.Linear(dim_in, dim_out), BatchNorm1d(dim_out))
        else:
            print('Linear projection layer without BN')

        # prediction head 
        if pred == 'mlpbn':
            print('MLP prediction layer with BN')
            self.pred = nn.Sequential(nn.Linear(dim_out, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out), BatchNorm1d(dim_out))
        elif pred == 'mlpbn1':
            print('MLP prediction layer with BN in the middle')
            self.pred = nn.Sequential(nn.Linear(dim_out, dim_mlp), BatchNorm1d(dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        elif pred == 'mlp':
            print('MLP prediction layer without BN')
            self.pred = nn.Sequential(nn.Linear(dim_out, dim_mlp), nn.ReLU(), nn.Linear(dim_mlp, dim_out))
        elif pred == 'linbn':
            print('Linear prediction layer with BN')
            self.pred = nn.Sequential(nn.Linear(dim_out, dim_out), BatchNorm1d(dim_out))
        elif pred == 'lin':
            print('Linear prediction layer without BN')
            self.pred = nn.Linear(dim_out, dim_out)
        else:
            self.pred = None

        # initialize the key encoder by the queue encoder
        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)
            param_k.requires_grad = False

        # queue for moco
        self.register_buffer("queue", torch.randn(dim, qlen))
        self.queue = nn.functional.normalize(self.queue, dim=0)

        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))
        
    
    @torch.no_grad()
    def _batch_shuffle_ddp(self, x):
        """
        Batch shuffle, for making use of BatchNorm.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # random shuffle index
        idx_shuffle = torch.randperm(batch_size_all).cuda()

        # broadcast to all gpus
        torch.distributed.broadcast(idx_shuffle, src=0)

        # index for restoring
        idx_unshuffle = torch.argsort(idx_shuffle)

        # shuffled index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_shuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this], idx_unshuffle

    @torch.no_grad()
    def _batch_unshuffle_ddp(self, x, idx_unshuffle):
        """
        Undo batch shuffle.
        *** Only support DistributedDataParallel (DDP) model. ***
        """
        # gather from all gpus
        batch_size_this = x.shape[0]
        x_gather = concat_all_gather(x)
        batch_size_all = x_gather.shape[0]

        num_gpus = batch_size_all // batch_size_this

        # restored index for this gpu
        gpu_idx = torch.distributed.get_rank()
        idx_this = idx_unshuffle.view(num_gpus, -1)[gpu_idx]

        return x_gather[idx_this]

    def forward(self, mode, im_1, im_2, flag, m=None, criterion=None, num_aux=0):

        bsz = im_1.shape[0]
        
        im1_rot = [torch.rot90(im_1, k, [2, 3]) for k in range(1,4)]  # 90, 180, 270 degrees
        im2_rot = [torch.rot90(im_2, k, [2, 3]) for k in range(1,4)]
   
        im1_rot = torch.stack(im1_rot)
        im2_rot = torch.stack(im2_rot)

        # symmetric loss
        if self.sym:
            im_qk = [(im_1, im_2, im2_rot), (im_2, im_1, im1_rot)]
        else:
            im_qk = [(im_1, im_2, im2_rot)]
        glogits = glabels = gloss = None

        for s, (im_q, im_k, im_rot) in enumerate(im_qk):
                       
            labels_aux = lam = None
            q = self.encoder_q(im_q)  # queries: NxC

            # prediction head and normalization
            if self.pred is not None:
                q = self.pred(q)
            q = nn.functional.normalize(q, dim=1)

            with torch.no_grad():  # no gradient to keys
                # update the key encoder
                if m is None: m = self.emam
                for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
                    param_k.data = param_k.data * m + param_q.data * (1. - m)

                # shuffle BN
                if self.shuffle_bn:
                    if self.distributed:
                        # make sure no grad in torch.distributed
                        with torch.no_grad():
                            im_k, idx_unshuffle_this = self._batch_shuffle_ddp(im_k)
                            k = self.encoder_k(im_k)  # keys: NxC
                            k = nn.functional.normalize(k, dim=1)
                            k = self._batch_unshuffle_ddp(k, idx_unshuffle_this)
                            ts = []
                            for m in range(3):
                                im_rot_m, ind_unshuffle_rot = self._batch_shuffle_ddp(im_rot[m, :].clone())
                                t = self.encoder_k(im_rot_m)
                                t = nn.functional.normalize(t, dim=1)
                                t = self._batch_unshuffle_ddp(t, ind_unshuffle_rot)
                                ts.append(t)
            # compute logits
            # positive logits: Nx1
            l_pos = torch.einsum('nc,nc->n', [q, k]).unsqueeze(-1)
            # negative logits: NxK
            # negative_rot logits: Nx3
            l_neg = torch.einsum('nc,ck->nk', [q, self.queue.clone().detach()])
            l_neg_rot = [torch.einsum('nc,nc->n', [q, t]).unsqueeze(-1)*1 for t in ts]
            if mode=='moco':
                logits = torch.cat([l_pos, l_neg], dim=1)  # logits: Nx(1+K)
            else:
                logits = torch.cat([l_pos, l_neg] + l_neg_rot, dim=1)  # logits: Nx(1+K+3)
                
            # apply temperature
            logits /= self.temp

            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            one_hot_labels = torch.nn.functional.one_hot(labels, num_classes=logits.shape[1]).to(torch.float64)
            posi_label = torch.zeros(logits.shape[1], dtype=torch.float64).cuda()
            posi_label[0] = 0.25
            posi_label[-3:] = 0.25 
            one_hot_labels[flag == 1] = posi_label  # create labels for PNDA
           
            # gather keys for original moco
            if self.distributed:
                k = concat_all_gather(k)
           
            bsz_all = k.shape[0]
            ptr = self.queue_ptr.item()
            assert self.qlen % bsz_all == 0, 'set qlen % batch_size == 0 for simpliclity'
            self.queue[:, ptr:ptr + bsz_all] = k.T
            ptr = (ptr + bsz_all) % self.qlen
            self.queue_ptr[0] = ptr

            loss = None
            assert s == 0  # not use symmetrized loss
            glogits = logits
            glabels = one_hot_labels
            gloss = loss
        return glogits, glabels, labels, gloss


@torch.no_grad()
def concat_all_gather(input):
    gathered = [torch.ones_like(input) for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(gathered, input, async_op=False)
    return torch.cat(gathered, dim=0)


class BatchNorm1d(nn.Module):
    def __init__(self, dim, affine=True, momentum=0.1):
        super(BatchNorm1d, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine, momentum=momentum)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


import os
import gc
import argparse
import numpy as np
import logging as logger

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn

from datasets import make_dataloader
from config import config as cfg
from models import make_model
from losses import SoftLoss, SP_KD_Loss
from utils import write_config_into_log, ramp_up, ramp_down


cfg.output_dir   = os.path.join(cfg.ckpt_root_dir, cfg.output_dir)
cfg.snapshot_dir = os.path.join(cfg.output_dir, 'snapshots')
cfg.train_log    = os.path.join(cfg.output_dir, 'train_log.txt')
cfg.test_log     = os.path.join(cfg.output_dir, 'test_log.txt')

print('Experiments dir: {}'.format(cfg.output_dir))
if not os.path.exists(cfg.output_dir):
    os.makedirs(cfg.output_dir)
if not os.path.exists(cfg.snapshot_dir):
    os.makedirs(cfg.snapshot_dir)

log_format = '%(levelname)s %(asctime)s %(filename)s] %(message)s'
logger.basicConfig(level=logger.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
fh = logger.FileHandler(cfg.train_log)
fh.setFormatter(logger.Formatter(log_format))
logger.getLogger().addHandler(fh)


def train():
    logger.info('Initializing....')
    cudnn.enable = True
    cudnn.benchmark = True
    # torch.manual_seed(1)
    # torch.cuda.manual_seed(1)
    write_config_into_log(cfg)

    logger.info('Building model......')
    if cfg.pretrained:
        model = make_model(cfg)
        model.load_param(cfg)
        logger.info('Loaded pretrained model from {0}'.format(cfg.pretrained))
    else:
        model = make_model(cfg)

    model.cuda()
    model = torch.nn.DataParallel(model)

    optimizer = torch.optim.Adam([{'params': model.module.base.parameters(), 'lr': cfg.get_lr(0)[0]},
                                  {'params': model.module.classifiers.parameters(), 'lr': cfg.get_lr(0)[1]}],
                                 weight_decay=cfg.weight_decay)
    
    celoss = nn.CrossEntropyLoss().cuda()
    softloss = SoftLoss()
    sp_kd_loss = SP_KD_Loss()
    criterions = [celoss, softloss, sp_kd_loss]

    train_loader, val_loader = make_dataloader(cfg)
    logger.info('Begin training......')
    for epoch in range(cfg.start_epoch, cfg.max_epoch):
        train_one_epoch(train_loader, val_loader, model, criterions, optimizer, epoch, cfg)

        total_acc = test(cfg, val_loader, model, epoch)
        with open(cfg.test_log, 'a+') as f:
            f.write('Epoch {0}: Acc is {1:.4f}\n'.format(epoch, total_acc))
        torch.save(obj=model.state_dict(),
                   f=os.path.join(cfg.snapshot_dir, 'ep{}_acc{:.4f}.pth'.format(epoch, total_acc)))
        logger.info('Model saved')


def train_one_epoch(train_loader, val_loader, model, criterions, optimizer, epoch, cfg):
    model.train()
    
    ramp_up_w, ramp_down_w = ramp_up(epoch, cfg.ramp_a), ramp_down(epoch, cfg.ramp_a)
    w, gamma = cfg.w, cfg.gamma
    logger.info('ramp_up_w: {:.4f}, ramp_down_w: {:.4f}, w: {}, gamma: {}'.format(ramp_up_w, ramp_down_w, w, gamma))
    adjust_learning_rate(optimizer, cfg.get_lr(epoch)[0], cfg.get_lr(epoch)[1])
    training_phase = cfg.train_mode

    for idx, batch in enumerate(train_loader, start=1):
        img, label = batch

        good_batch = check_batch(label, cfg.num_classes)
        if not good_batch:
            continue
        img, label = img.cuda(), label.cuda()
        
        x_final, output_x_list, targets, softlabel_list, G_matrixs, G_main_matrixs, score, atten_x_final = model(img, label, training_phase)
        
        softlabel = torch.zeros_like(x_final)
        for c in range(cfg.num_classes):
            # sharpen
            px = torch.softmax(softlabel_list[c], dim=1)
            ptx = px ** (1 / cfg.T)  # temparature sharpening
            targets_x = ptx / ptx.sum(dim=1, keepdim=True)  # normalize
            targets_x = targets_x.detach()

            _h, _w = px.shape
            _targets = torch.zeros([_h, _w+1]).float()
            _targets[:, 0:c], _targets[:, c + 1:] = targets_x[:, 0:c], targets_x[:, c:]
            ind = (label == c).nonzero()
            softlabel[ind[:, 0]] = _targets.cuda() # bs x num_cls
        
        softLoss = criterions[1](x_final, label, softlabel)
        aux_loss = [criterions[0](_pred, _label) for _pred, _label in zip(output_x_list, targets)] # auxiliary branch  
        aux_loss = sum(aux_loss) / (cfg.num_branches-1)
        CEloss = criterions[0](atten_x_final, label) # main
        spLoss = criterions[2](G_matrixs, G_main_matrixs)

        loss = ramp_up_w * CEloss + ramp_up_w * (w * softLoss + gamma * spLoss) +  ramp_down_w * aux_loss
            
        if idx % 20 == 0:
            logger.info('Epoch {} Batch {}/{}: CEloss {:.6f}, softLoss {:.6f}, spLoss {:.6f}, aux_loss {:.6f}'.format(
                            epoch, idx, len(train_loader), CEloss, softLoss, spLoss, aux_loss))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if idx % 200 == 0:
            total_acc = test(cfg, val_loader, model, epoch, idx)
            with open(cfg.test_log, 'a+') as f:
                f.write('Epoch {0} Batch {1}: Acc is {2:.4f}\n'.format(epoch, idx, total_acc))
            torch.save(obj=model.state_dict(),
                       f=os.path.join(cfg.snapshot_dir, 'ep{}_b{}_acc{:.4f}.pth'.format(epoch, idx, total_acc)))
            logger.info('Model saved')

        gc.collect()


def test_worker(val_loader, model):
    Pred, Label = [], []
    for idx, batch in enumerate(val_loader, 1):
        img, label = batch
        pred, output_x_list, targets, softlabel_list = model(img.cuda(), label, 'normal')

        for i in range(pred.shape[0]):
            x = pred[i].data.cpu().numpy()
            y = label[i].item()
            Pred.append(x)
            Label.append(y)

    return Pred, Label


def test(cfg, val_loader, model, epoch, batch=None):
    model.eval()

    pred, label = test_worker(val_loader, model)
    pred = [np.where(x == np.max(x))[0][0] for x in pred]

    total = len([x for x in range(len(pred)) if pred[x] == label[x]])
    total_acc = total / len(pred)
    logger.info('Epoch {}: Acc is {:.4f}'.format(epoch, total_acc)) if batch is None else \
        logger.info('Epoch {} Batch {}: Acc is {:.4f}'.format(epoch, batch, total_acc))

    return total_acc


def adjust_learning_rate(optimizer, lr1, lr2):
    optimizer.param_groups[0]['lr'] = lr1
    optimizer.param_groups[1]['lr'] = lr2


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def check_batch(label, num_classes):
    for i in range(num_classes):
        cnt = (label == i).nonzero().shape[0]
        if cnt < 2:
            return False
    return True


if __name__ == '__main__':
    train()

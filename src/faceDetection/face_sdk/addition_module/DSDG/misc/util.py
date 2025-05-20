import os
import time
import argparse
import numpy as np

import torch
import torch.nn.functional as F


def rgb2gray(img):
    r, g, b = torch.split(img, 1, dim=1)
    return torch.mul(r, 0.299) + torch.mul(g, 0.587) + torch.mul(b, 0.114)


def reparameterize(mu, logvar):
    std = logvar.mul(0.5).exp_()
    eps = torch.cuda.FloatTensor(std.size()).normal_()
    return eps.mul(std).add_(mu)


def kl_loss(mu, logvar, prior_mu=0):
    v_kl = mu.add(-prior_mu).pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
    v_kl = v_kl.sum(dim=-1).mul_(-0.5)  # (batch, 2)
    return v_kl


def reconstruction_loss(prediction, target, size_average=False):
    error = (prediction - target).view(prediction.size(0), -1)
    error = error ** 2
    error = torch.sum(error, dim=-1)

    if size_average:
        error = error.mean()
    else:
        error = error.sum()
    return error


def load_model(model, pretrained):
    weights = torch.load(pretrained)
    pretrained_dict = weights['model'].state_dict()
    model_dict = model.state_dict()
    # 1. filter out unnecessary keys
    pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
    # 2. overwrite entries in the existing state dict
    model_dict.update(pretrained_dict)
    # 3. load the new state dict
    model.load_state_dict(model_dict)


def save_checkpoint(model_path, model, epoch, iteration, name):
    model_out_path = model_path + name + "model_epoch_{}_iter_{}.pth".format(epoch, iteration)
    state = {"epoch": epoch, "model": model}
    if not os.path.exists(model_path):
        os.makedirs(model_path)
    torch.save(state, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def MMD_Loss(fc_nir, fc_vis):
    mean_fc_nir = torch.mean(fc_nir, 0)
    mean_fc_vis = torch.mean(fc_vis, 0)
    loss_mmd = F.mse_loss(mean_fc_nir, mean_fc_vis)
    return loss_mmd


def adjust_learning_rate(lr, step, optimizer, epoch):
    scale = 0.457305051927326
    lr = lr * (scale ** (epoch // step))
    print('lr: {}'.format(lr))
    if (epoch != 0) & (epoch % step == 0):
        print('Change lr')
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * scale


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

import sys
import math
import torch
import torch.nn.functional as F
from torch.nn import Module, Parameter
import torch.nn as nn

from backbones.resnet import ResNet, BasicBlock, Bottleneck
from backbones.resnet_ibn_a import resnet50_ibn_a
# from backbones.attention_irse_def import AttentionNet
# from backbones.resnet_irse_def import Backbone
# from backbones.mobilefacenet_def import MobileFaceNet

def l2_norm(input, axis=1):
    norm = torch.norm(input, 2, axis, True)
    output = input / norm
    return output

#  Arcface head #
class ArcMarginProduct(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, num_class=86876, s=64., m=0.5, easy_margin=True):
        super(ArcMarginProduct, self).__init__()
        self.num_class = num_class
        self.weight = Parameter(torch.Tensor(embedding_size, num_class))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = m  # the margin value, default is 0.5
        self.s = s  # scalar value default is 64, see normface https://arxiv.org/abs/1704.06369
        self.cos_m = math.cos(m)
        self.sin_m = math.sin(m)
        self.mm = self.sin_m * m  # issue 1
        self.threshold = math.cos(math.pi - m)
        self.easy_margin = easy_margin

    def forward(self, embeddings, label):

        kernel_norm = l2_norm(self.weight, axis=0)
        # cos(theta + m) = cos(theta)cos(m) - sin(theta)sin(m)
        cos_theta = torch.mm(embeddings, kernel_norm)
        #         output = torch.mm(embbedings,kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability

        sin_theta = torch.sqrt(1.0 - torch.pow(cos_theta, 2))
        cos_theta_m = (cos_theta * self.cos_m - sin_theta * self.sin_m)
        # this condition controls the theta+m should in range [0, pi]
        #      0<=theta+m<=pi
        #     -m<=theta<=pi-m
        if self.easy_margin:
            cos_theta_m = torch.where(cos_theta > 0, cos_theta_m, cos_theta)
        else:
            cos_theta_m = torch.where(cos_theta > self.threshold, cos_theta_m, cos_theta - self.mm)

        # label = label.view(-1, 1)  # size=(B,1)
        output = cos_theta * 1.0  # a little bit hacky way to prevent in_place operation on cos_theta
        batch_size = label.size(0)
        output[torch.arange(0, batch_size), label] = cos_theta_m[torch.arange(0, batch_size), label]
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output


# Cosface head #
class Am_softmax(Module):
    # implementation of additive margin softmax loss in https://arxiv.org/abs/1801.05599
    def __init__(self, embedding_size=512, classnum=51332):
        super(Am_softmax, self).__init__()
        self.classnum = classnum
        self.kernel = Parameter(torch.Tensor(embedding_size, classnum))
        # initial kernel
        self.kernel.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.m = 0.35  # additive margin recommended by the paper
        self.s = 30.  # see normface https://arxiv.org/abs/1704.06369

    def forward(self, embbedings, label):
        kernel_norm = l2_norm(self.kernel, axis=0)
        cos_theta = torch.mm(embbedings, kernel_norm)
        cos_theta = cos_theta.clamp(-1, 1)  # for numerical stability
        phi = cos_theta - self.m
        label = label.view(-1, 1)  # size=(B,1)
        index = cos_theta.data * 0.0  # size=(B,Classnum)
        index.scatter_(1, label.data.view(-1, 1), 1)
        index = index.byte()
        output = cos_theta * 1.0
        output[index] = phi[index]  # only change the correct predicted output
        output *= self.s  # scale up in order to make softmax work, first introduced in normface
        return output

# MV-Softmax
class MV_Softmax(Module):
    def __init__(self, feat_dim, num_class, is_am, margin=0.35, mask=1.12, scale=32):
        super(MV_Softmax, self).__init__()
        self.weight = Parameter(torch.Tensor(feat_dim, num_class))
        # initial kernel
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)
        self.margin = margin
        self.mask = mask
        self.scale = scale
        self.is_am = is_am
        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.threshold = math.cos(math.pi - margin)
        self.mm = self.sin_m * margin

    def forward(self, x, label):  # x (M, K), w(K, N), y = xw (M, N), note both x and w are already l2 normalized.
        kernel_norm = F.normalize(self.weight, dim=0)
        cos_theta = torch.mm(x, kernel_norm)
        batch_size = label.size(0)

        gt = cos_theta[torch.arange(0, batch_size), label].view(-1, 1)  # get ground truth score of cos distance
        if self.is_am:  # AM
            mask = cos_theta > gt - self.margin
            final_gt = torch.where(gt > self.margin, gt - self.margin, gt)
        else:  # arcface
            sin_theta = torch.sqrt(1.0 - torch.pow(gt, 2))
            cos_theta_m = gt * self.cos_m - sin_theta * self.sin_m  # cos(gt + margin)
            mask = cos_theta > cos_theta_m
            final_gt = torch.where(gt > 0.0, cos_theta_m, gt)
        # process hard example.
        hard_example = cos_theta[mask]
        cos_theta[mask] = self.mask * hard_example + self.mask - 1.0
        cos_theta.scatter_(1, label.data.view(-1, 1), final_gt)
        cos_theta *= self.scale
        return cos_theta


class MVFace(torch.nn.Module):
    def __init__(self, backbone, feat_dim, num_class, is_am):
        super(MVFace, self).__init__()

        if backbone == 'resnet18':
            self.feat_net = ResNet(last_stride=2,
                            block=BasicBlock, frozen_stages=-1,
                            layers=[2, 2, 2, 2])
        elif backbone == 'resnet50':
            self.feat_net = ResNet(last_stride=2,
                            block=Bottleneck, frozen_stages=-1,
                            layers=[3, 4, 6, 3])
        elif backbone == 'resnet50_ibn':
            self.feat_net = resnet50_ibn_a(last_stride=2)
        else:
            raise Exception('backbone must be resnet18, resnet50 and resnet50_ibn.')

        self.gap = nn.AdaptiveAvgPool2d(1)
        self.loss_layer = MV_Softmax(feat_dim=feat_dim, num_class=num_class, is_am=is_am)

    def forward(self, data, label):
        feat = self.feat_net.forward(data)
        feat = self.gap(feat).squeeze()
        pred = self.loss_layer.forward(feat, label)

        return pred

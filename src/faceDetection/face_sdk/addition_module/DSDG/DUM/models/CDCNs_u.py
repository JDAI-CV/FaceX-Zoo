import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np


class Conv2d_cd(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, groups=1, bias=False, theta=0.7):

        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding,
                              dilation=dilation, groups=groups, bias=bias)
        self.theta = theta

    def forward(self, x):
        out_normal = self.conv(x)

        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal
        else:
            # pdb.set_trace()
            [C_out, C_in, kernel_size, kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = F.conv2d(input=x, weight=kernel_diff, bias=self.conv.bias, stride=self.conv.stride, padding=0,
                                groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class SpatialAttention(nn.Module):
    def __init__(self, kernel=3):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)

        return self.sigmoid(x)


class CDCN_u(nn.Module):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.7):
        super(CDCN_u, self).__init__()

        self.conv1 = nn.Sequential(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1 = nn.Sequential(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 196, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            basic_conv(196, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1 = nn.Sequential(
            basic_conv(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            basic_conv(128, 64, kernel_size=3, stride=1, padding=1, bias=False, theta=theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.mu_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.logvar_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):  # x [3, 256, 256]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)  # x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)  # x [128, 32, 32]

        x_Block2 = self.Block2(x_Block1)  # x [128, 64, 64]
        x_Block2_32x32 = self.downsample32x32(x_Block2)  # x [128, 32, 32]

        x_Block3 = self.Block3(x_Block2)  # x [128, 32, 32]
        x_Block3_32x32 = self.downsample32x32(x_Block3)  # x [128, 32, 32]

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)  # x [128*3, 32, 32]

        # pdb.set_trace()

        x = self.lastconv1(x_concat)  # x [128, 32, 32]
        x = self.lastconv2(x)  # x [64, 32, 32]

        mu = self.mu_head(x)
        mu = mu.squeeze(1)
        logvar = self.logvar_head(x)
        logvar = logvar.squeeze(1)
        embedding = self._reparameterize(mu, logvar)

        return mu, logvar, embedding, x_concat, x_Block1, x_Block2, x_Block3, x_input


class depthnet_u(nn.Module):

    def __init__(self):
        super(depthnet_u, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.Block1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),

        )

        self.Block2 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.Block3 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 196, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(196),
            nn.ReLU(),
            nn.Conv2d(196, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
        )

        self.lastconv1 = nn.Sequential(
            nn.Conv2d(128 * 3, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )

        self.lastconv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )

        self.mu_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.logvar_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
        )

        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):  # x [3, 256, 256]

        x_input = x
        x = self.conv1(x)

        x_Block1 = self.Block1(x)  # x [128, 128, 128]
        x_Block1_32x32 = self.downsample32x32(x_Block1)  # x [128, 32, 32]

        x_Block2 = self.Block2(x_Block1)  # x [128, 64, 64]
        x_Block2_32x32 = self.downsample32x32(x_Block2)  # x [128, 32, 32]

        x_Block3 = self.Block3(x_Block2)  # x [128, 32, 32]
        x_Block3_32x32 = self.downsample32x32(x_Block3)  # x [128, 32, 32]

        x_concat = torch.cat((x_Block1_32x32, x_Block2_32x32, x_Block3_32x32), dim=1)  # x [128*3, 32, 32]

        # pdb.set_trace()

        x = self.lastconv1(x_concat)  # x [128, 32, 32]
        x = self.lastconv2(x)  # x [64, 32, 32]

        mu = self.mu_head(x)
        mu = mu.squeeze(1)
        logvar = self.logvar_head(x)
        logvar = logvar.squeeze(1)
        embedding = self._reparameterize(mu, logvar)

        return mu, logvar, embedding, x_concat, x_Block1, x_Block2, x_Block3, x_input
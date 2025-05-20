import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class Encoder(nn.Module):
    def __init__(self, hdim=256):
        super(Encoder, self).__init__()

        self.hdim = hdim

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 32, 64),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 64, 128),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 128, 256),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 256, 512),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 512, 512)
        )

        # mu and logvar
        self.fc = nn.Linear(512 * 4 * 4, 2 * hdim)

    def forward(self, x):
        z = self.main(x).view(x.size(0), -1)
        z = self.fc(z)
        mu, logvar = torch.split(z, split_size_or_sections=self.hdim, dim=-1)

        return mu, logvar


class Encoder_s(nn.Module):
    def __init__(self, hdim=256):
        super(Encoder_s, self).__init__()

        self.hdim = hdim

        self.main = nn.Sequential(
            nn.Conv2d(3, 32, 5, 2, 2, bias=False),
            nn.InstanceNorm2d(32, eps=0.001),
            nn.LeakyReLU(0.2),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 32, 64),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 64, 128),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 128, 256),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 256, 512),
            nn.AvgPool2d(2),
            make_layer(_Residual_Block, 1, 512, 512)
        )

        # mu and logvar
        self.fc = nn.Linear(512 * 4 * 4, 4 * hdim)
        # self.fc_at = nn.Linear(hdim, attack_type)

    def forward(self, x):
        z = self.main(x).view(x.size(0), -1)
        z = self.fc(z)
        mu, logvar, mu_a, logvar_a = torch.split(z, split_size_or_sections=self.hdim, dim=-1)
        # a_type = self.fc_at(a_t)

        return mu, logvar, mu_a, logvar_a


class Cls(nn.Module):
    def __init__(self, hdim=256, attack_type=4):
        super(Cls, self).__init__()

        self.fc = nn.Linear(hdim, attack_type)

    def forward(self, x):
        a_cls = self.fc(x)

        return a_cls

class Decoder(nn.Module):
    def __init__(self, hdim=256):
        super(Decoder, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(hdim, 512 * 4 * 4),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            make_layer(_Residual_Block, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 1, 512, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 1, 256, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 2, 128, 64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3 + 3, 5, 1, 2)
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        x = y.view(z.size(0), -1, 4, 4)
        img = torch.sigmoid(self.main(x))
        return img


class Decoder_s(nn.Module):
    def __init__(self, hdim=256):
        super(Decoder_s, self).__init__()

        self.fc = nn.Sequential(
            nn.Linear(3 * hdim, 512 * 4 * 4),
            nn.ReLU(True)
        )

        self.main = nn.Sequential(
            make_layer(_Residual_Block, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 1, 512, 512),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 1, 512, 256),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 1, 256, 128),
            nn.Upsample(scale_factor=2, mode='nearest'),
            make_layer(_Residual_Block, 2, 128, 64),
            nn.Upsample(scale_factor=2, mode='nearest'),
            nn.Conv2d(64, 3 + 3, 5, 1, 2)
        )

    def forward(self, z):
        z = z.view(z.size(0), -1)
        y = self.fc(z)
        x = y.view(z.size(0), -1, 4, 4)
        img = torch.sigmoid(self.main(x))
        return img


class _Residual_Block(nn.Module):
    def __init__(self, inc=64, outc=64, groups=1):
        super(_Residual_Block, self).__init__()

        if inc is not outc:
            self.conv_expand = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=1, stride=1, padding=0,
                                         groups=1, bias=False)
        else:
            self.conv_expand = None

        self.conv1 = nn.Conv2d(in_channels=inc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn1 = nn.InstanceNorm2d(outc, eps=0.001)
        self.relu1 = nn.LeakyReLU(0.2, inplace=True)
        self.conv2 = nn.Conv2d(in_channels=outc, out_channels=outc, kernel_size=3, stride=1, padding=1, groups=groups,
                               bias=False)
        self.bn2 = nn.InstanceNorm2d(outc, eps=0.001)
        self.relu2 = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        if self.conv_expand is not None:
            identity_data = self.conv_expand(x)
        else:
            identity_data = x

        output = self.relu1(self.bn1(self.conv1(x)))
        output = self.conv2(output)
        output = self.relu2(self.bn2(torch.add(output, identity_data)))
        return output


def make_layer(block, num_of_layer, inc=64, outc=64, groups=1):
    if num_of_layer < 1:
        num_of_layer = 1
    layers = []
    layers.append(block(inc=inc, outc=outc, groups=groups))
    for _ in range(1, num_of_layer):
        layers.append(block(inc=outc, outc=outc, groups=groups))
    return nn.Sequential(*layers)

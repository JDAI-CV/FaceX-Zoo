import math

import torch
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch import nn
from torch.nn import Parameter
import pdb
import numpy as np


class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, ResidualBlock = ResidualBlock):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  2, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 2, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 196, 2, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 256, 2, stride=2)
        self.lastconv1 = nn.Sequential(
            nn.Conv2d(128 + 196 + 256, 128, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(),
        )
        self.lastconv2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.lastconv3 = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            nn.ReLU(),
        )
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')
        
        self.mu_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            
        )
        self.logvar_head = nn.Sequential(
            nn.Conv2d(64, 1, kernel_size=3, stride=1, padding=1, bias=False),
            
        )

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.Sequential(*layers)

    def _reparameterize(self, mu, logvar):
        std = torch.exp(logvar).sqrt()
        epsilon = torch.randn_like(std)
        return mu + epsilon * std

    def forward(self, x):
        x_input = x
        x = self.conv1(x)
        x_block1 = self.layer1(x)
        x_block2 = self.layer2(x_block1)
        x_block2_32 = self.downsample32x32(x_block2)
        x_block3 = self.layer3(x_block2)
        x_block3_32 = self.downsample32x32(x_block3)
        x_block4 = self.layer4(x_block3)
        x_block4_32 = self.downsample32x32(x_block4)

        x_concat = torch.cat((x_block2_32, x_block3_32, x_block4_32), dim=1)

        x = self.lastconv1(x_concat)
        x = self.lastconv2(x)
        mu = self.mu_head(x)
        mu = mu.squeeze(1)
        logvar = self.logvar_head(x)
        logvar = logvar.squeeze(1)
        embedding = self._reparameterize(mu, logvar)

        return mu, logvar, embedding, x_concat, x_block2, x_block3, x_block4, x_input


def ResNet18_u():

    return ResNet(ResidualBlock)


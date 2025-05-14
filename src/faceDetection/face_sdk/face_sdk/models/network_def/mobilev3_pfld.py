# derive from:
# https://github.com/Hsintao/pfld_106_face_landmarks/blob/master/models/mobilev3_pfld.py

import torch
import torch.nn as nn
import torch.nn.functional as F


def conv_bn(inp, oup, kernel_size, stride, padding=1, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, kernel_size, stride, padding, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


def conv_1x1_bn(inp, oup, conv_layer=nn.Conv2d, norm_layer=nn.BatchNorm2d, nlin_layer=nn.ReLU):
    return nn.Sequential(
        conv_layer(inp, oup, 1, 1, 0, bias=False),
        norm_layer(oup),
        nlin_layer(inplace=True)
    )


class Hswish(nn.Module):
    def __init__(self, inplace=True):
        super(Hswish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return x * F.relu6(x + 3., inplace=self.inplace) / 6.


class Hsigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(Hsigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return F.relu6(x + 3., inplace=self.inplace) / 6.


class SEModule(nn.Module):
    def __init__(self, channel, reduction=4):
        super(SEModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            Hsigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.size()
        # F.avg_pool2d()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


class Identity(nn.Module):
    def __init__(self, channel):
        super(Identity, self).__init__()

    def forward(self, x):
        return x


class MobileBottleneck(nn.Module):
    def __init__(self, inp, oup, kernel, stride, exp, se=False, nl='RE'):
        super(MobileBottleneck, self).__init__()
        assert stride in [1, 2]
        assert kernel in [3, 5]
        padding = (kernel - 1) // 2
        self.use_res_connect = stride == 1 and inp == oup

        conv_layer = nn.Conv2d
        norm_layer = nn.BatchNorm2d
        if nl == 'RE':
            nlin_layer = nn.ReLU  # or ReLU6
        elif nl == 'HS':
            nlin_layer = Hswish
        else:
            raise NotImplementedError
        if se:
            SELayer = SEModule
        else:
            SELayer = Identity

        self.conv = nn.Sequential(
            # pw
            conv_layer(inp, exp, 1, 1, 0, bias=False),
            norm_layer(exp),
            nlin_layer(inplace=True),
            # dw
            conv_layer(exp, exp, kernel, stride, padding, groups=exp, bias=False),
            norm_layer(exp),
            SELayer(exp),
            nlin_layer(inplace=True),
            # pw-linear
            conv_layer(exp, oup, 1, 1, 0, bias=False),
            norm_layer(oup),
        )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class PFLDInference(nn.Module):
    def __init__(self):
        super(PFLDInference, self).__init__()
        self.use_attention = True
        self.conv_bn1 = conv_bn(3, 16, 3, stride=1, nlin_layer=Hswish)
        self.conv_bn2 = MobileBottleneck(16, 16, 3, 1, 16, False, 'RE')

        self.conv3_1 = MobileBottleneck(16, 24, 3, 2, 64, False, 'RE')

        self.block3_2 = MobileBottleneck(24, 24, 3, 1, 72, False, "RE")
        self.block3_3 = MobileBottleneck(24, 40, 5, 2, 72, self.use_attention, "RE")
        self.block3_4 = MobileBottleneck(40, 40, 5, 1, 120, self.use_attention, "RE")
        self.block3_5 = MobileBottleneck(40, 40, 5, 1, 120, self.use_attention, "RE")

        self.conv4_1 = MobileBottleneck(40, 80, 3, 2, 240, False, "RE")

        self.conv5_1 = MobileBottleneck(80, 80, 3, 1, 200, False, "HS")
        self.block5_2 = MobileBottleneck(80, 112, 3, 1, 480, self.use_attention, "HS")
        self.block5_3 = MobileBottleneck(112, 112, 3, 1, 672, self.use_attention, "HS")
        self.block5_4 = MobileBottleneck(112, 160, 3, 1, 672, self.use_attention, "HS")

        self.conv6_1 = MobileBottleneck(160, 16, 3, 1, 320, False, "HS")  # [16, 14, 14]

        self.conv7 = nn.Conv2d(16, 32, 3, 2, padding=1)
        self.conv8 = nn.Conv2d(32, 128, 7, 1, 0)
        self.avg_pool1 = nn.AvgPool2d(14)
        self.avg_pool2 = nn.AvgPool2d(7)
        self.fc = nn.Linear(176, 106 * 2)

    def forward(self, x):  # x: 3, 112, 112
        x = self.conv_bn1(x)  # [64, 56, 56]
        x = self.conv_bn2(x)  # [64, 56, 56]
        x = self.conv3_1(x)
        x = self.block3_2(x)
        x = self.block3_3(x)
        x = self.block3_4(x)
        out1 = self.block3_5(x)

        x = self.conv4_1(out1)

        x = self.conv5_1(x)
        x = self.block5_2(x)
        x = self.block5_3(x)
        x = self.block5_4(x)
        x = self.conv6_1(x)
        x1 = self.avg_pool1(x)
        x1 = x1.view(x1.size(0), -1)

        x = self.conv7(x)
        x2 = self.avg_pool2(x)
        x2 = x2.view(x2.size(0), -1)

        x3 = self.conv8(x)
        x3 = x3.view(x1.size(0), -1)

        multi_scale = torch.cat([x1, x2, x3], 1)
        landmarks = self.fc(multi_scale)

        return out1, landmarks


class AuxiliaryNet(nn.Module):
    def __init__(self):
        super(AuxiliaryNet, self).__init__()
        self.conv1 = conv_bn(40, 128, 3, 2)
        self.conv2 = conv_bn(128, 128, 3, 1)
        self.conv3 = conv_bn(128, 32, 3, 2)
        self.conv4 = conv_bn(32, 128, 3, 1, padding=0)
        self.max_pool1 = nn.MaxPool2d(5)
        self.fc1 = nn.Linear(128, 32)
        self.fc2 = nn.Linear(32, 3)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.max_pool1(x)
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)

        return x

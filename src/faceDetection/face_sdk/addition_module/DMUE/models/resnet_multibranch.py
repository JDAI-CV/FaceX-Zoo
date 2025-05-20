import math
import torch
from torch import nn


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(self, num_branches, num_classes,
                 last_stride=2, block=Bottleneck, frozen_stages=-1,layers=[3, 4, 6, 3]):
        self.inplanes = 64
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=None, padding=0)
        self.frozen_stages = frozen_stages
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        
        fix_inplanes = self.inplanes
        self.num_branches = num_branches
        self.num_classes = num_classes
        for i in range(self.num_branches):
            setattr(self, 'layer4_'+str(i), self._make_layer(block, 512, layers[3], stride=last_stride))
            self.inplanes = fix_inplanes

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, target, training_phase, c=None):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x_list = []
        softlabel_list = []
        x_final = None
        targets = None

        # auxiliary & main branch forward
        features, targets, inds = self.split_feature(x, target)
        for i in range(self.num_branches-1):
            x_list.append(getattr(self, 'layer4_'+str(i))(features[i]))
        x_final = getattr(self, 'layer4_'+str(self.num_branches-1))(x)
        # make label distribution
        features2, inds_softlabel = self.split_feature_makeLD(x, target)
        for i in range(self.num_branches-1):
            softlabel_list.append(getattr(self, 'layer4_'+str(i))(features2[i]))

        return x_final, x_list, softlabel_list, targets, inds, inds_softlabel

    def split_feature(self, x, target):
        x_parts = []
        t_parts = []
        inds = []
        for c in range(self.num_classes):
            ind = (target != c).nonzero()
            inds.append(ind)
            _t = target[ind[:,0]]
            t_parts.append(_t.where(_t < c, _t-1))
            x_parts.append(x[ind[:, 0], :, :, :])
        return x_parts, t_parts, inds

    def split_feature_makeLD(self, x, target):
        x_parts = []
        inds = []
        for c in range(self.num_classes):
            ind = (target == c).nonzero()
            inds.append(ind)
            x_parts.append(x[ind[:, 0], :, :, :])
        return x_parts, inds

    def random_init(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _freeze_stages(self):
        if self.frozen_stages >= 0:
            self.bn1.eval()
            for m in [self.conv1, self.bn1]:
                for param in m.parameters():
                    param.requires_grad = False

        for i in range(1, self.frozen_stages + 1):
            m = getattr(self, 'layer{}'.format(i))
            # print('layer{}'.format(i))
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

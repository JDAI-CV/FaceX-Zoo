import torch
import torch.nn as nn
from .resnet import ResNet, BasicBlock, Bottleneck
from .resnet_ibn_a import ResNet_IBN, Bottleneck_IBN


class Backbone(nn.Module):
    def __init__(self, cfg):
        super(Backbone, self).__init__()
        last_stride = cfg.last_stride
        model_name = cfg.backbone
        self.num_classes = cfg.num_classes
        self.num_branches = cfg.num_branches

        # base model
        if model_name == 'resnet18':
            self.in_planes = 512
            self.base = ResNet(last_stride=last_stride, block=BasicBlock, layers=[2, 2, 2, 2])
        elif model_name == 'resnet50_ibn':
            self.in_planes = 2048
            self.base = ResNet_IBN(last_stride=last_stride, block=Bottleneck_IBN, layers=[3, 4, 6, 3])
        else:
            raise ValueError('Unsupported backbone! but got {}'.format(model_name))

        # pooling after base
        self.gap = nn.AdaptiveAvgPool2d(1)

        # loss
        self.classifiers = []
        self.classifiers.append(nn.Linear(self.in_planes, self.num_classes, bias=cfg.BiasInCls))
        self.classifiers = nn.ModuleList(self.classifiers)
        
        self.neck = cfg.bnneck
        # print('Use bnneck', self.neck)
        if cfg.bnneck:
            self.bottleneck = nn.BatchNorm1d(self.in_planes)


    def forward(self, x):
        x_final = self.base(x)
        x_final = self.gap(x_final).squeeze(2).squeeze(2)
        x_final = self.bottleneck(x_final) if self.neck else x_final
        x_final = self.classifiers[0](x_final)
                              
        return x_final

    def load_param(self, cfg):
        pretrained = cfg.pretrained
        param_dict = torch.load(pretrained, map_location=lambda storage,loc: storage.cpu())

        if cfg.pretrained_choice == '':
            for i in param_dict:
                if 'fc.' in i: continue
                self.state_dict()[i].copy_(param_dict[i])
        elif cfg.pretrained_choice == 'convert':
            for i in param_dict:
                if 'layer4' in i:
                    if 'layer4_{}'.format(self.num_branches-1) in i:
                        j = i.replace('module.', '')
                        j = j.replace('layer4_{}'.format(self.num_branches-1), 'layer4')
                        self.state_dict()[j].copy_(param_dict[i])
                    else:
                        pass
                elif 'project_w' in i:
                    pass
                elif 'classifiers' in i:
                    if 'classifiers.{}'.format(self.num_branches-1) in i:
                        j = i.replace('module.', '')
                        j = j.replace('{}'.format(self.num_branches-1), '0')
                        self.state_dict()[j].copy_(param_dict[i])
                    else:
                        pass
                else:
                    j = i.replace('module.', '')
                    self.state_dict()[j].copy_(param_dict[i])
        else:
            raise ValueError('Unsupported pretrained_choice!')


def make_target_model(cfg):
    model = Backbone(cfg)
    return model

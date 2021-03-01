"""
@author: Jun Wang
@date: 20210301
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/zhanghang1989/ResNeSt/blob/master/resnest/torch/resnest.py

import torch
import torch.nn as nn
from .resnet import ResNet, Bottleneck

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)

def l2_norm(input,axis=1):
    norm = torch.norm(input,2,axis,True)
    output = torch.div(input, norm)
    return output
                        
class ResNeSt(nn.Module):
    def __init__(self, num_layers, drop_ratio, feat_dim, out_h=7, out_w=7):
        super(ResNeSt, self).__init__()
        self.input_layer = nn.Sequential(nn.Conv2d(3, 64, (3, 3), 1, 1 ,bias=False),
                                      nn.BatchNorm2d(64),
                                      nn.PReLU(64))
        self.output_layer = nn.Sequential(nn.BatchNorm2d(2048),
                                       nn.Dropout(drop_ratio),
                                       Flatten(),
                                       nn.Linear(2048 * out_h * out_w, feat_dim),
                                       nn.BatchNorm1d(feat_dim))
        if num_layers == 50:
            self.body = ResNet(Bottleneck, [3, 4, 6, 3],
                                       radix=2, groups=1, bottleneck_width=64,
                                       deep_stem=True, stem_width=32, avg_down=True,
                                       avd=True, avd_first=False)
        elif num_layers == 101:
            self.body = ResNet(Bottleneck, [3, 4, 23, 3],
                               radix=2, groups=1, bottleneck_width=64,
                               deep_stem=True, stem_width=64, avg_down=True,
                               avd=True, avd_first=False)
        elif num_layers == 200:
            self.body = ResNet(Bottleneck, [3, 24, 36, 3],
                               radix=2, groups=1, bottleneck_width=64,
                               deep_stem=True, stem_width=64, avg_down=True,
                               avd=True, avd_first=False)
        elif num_layers == 269:
            self.body = ResNet(Bottleneck, [3, 30, 48, 8],
                               radix=2, groups=1, bottleneck_width=64,
                               deep_stem=True, stem_width=64, avg_down=True,
                               avd=True, avd_first=False)
        else:
            pass
    def forward(self, x):
        x = self.input_layer(x)
        x = self.body(x)
        x = self.output_layer(x)
        return l2_norm(x)

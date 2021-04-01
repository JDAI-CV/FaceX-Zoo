"""
@author: Jun Wang 
@date: 20201019 
@contact: jun21wangustc@gmail.com    
"""

import sys
import yaml
sys.path.append('../../')
from backbone.ResNets import Resnet
from backbone.MobileFaceNets import MobileFaceNet

class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type(str): which backbone will produce.
        backbone_param(dict):  parsed params and it's value. 
    """
    def __init__(self, backbone_conf_file):
        with open(backbone_conf_file) as f:
            self.backbone_conf = yaml.load(f)

    def get_backbone(self, backbone_type):
        if backbone_type == 'MobileFaceNet':
            backbone_param = self.backbone_conf[backbone_type]
            feat_dim = backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = backbone_param['out_h'] # height of the feature map before the final features.
            out_w = backbone_param['out_w'] # width of the feature map before the final features.
            backbone = MobileFaceNet(feat_dim, out_h, out_w)
        elif 'ResNet' in backbone_type:
            backbone_param = self.backbone_conf[backbone_type]
            depth = backbone_param['depth'] # depth of the ResNet, e.g. 50, 100, 152.
            drop_ratio = backbone_param['drop_ratio'] # drop out ratio.
            net_mode = backbone_param['net_mode'] # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
            feat_dim = backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = backbone_param['out_h'] # height of the feature map before the final features.
            out_w = backbone_param['out_w'] # width of the feature map before the final features.
            backbone = Resnet(depth, drop_ratio, net_mode, feat_dim, out_h, out_w)
        else:
            pass
        return backbone

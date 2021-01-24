"""
@author: Jun Wang 
@date: 20201012 
@contact: jun21wangustc@gmail.com
"""

import sys
import torch
from thop import profile
from thop import clever_format

sys.path.append('..')
from backbone.backbone_def import BackboneFactory

#backbone_type = 'MobileFaceNet'
#backbone_type = 'ResNet'
#backbone_type = 'EfficientNet'
#backbone_type = 'HRNet'
backbone_type = 'GhostNet'
#backbone_type = 'AttentionNet'
#backbone_type = 'TF-NAS'

backbone_conf_file = '../training_mode/backbone_conf.yaml'
backbone_factory = BackboneFactory(backbone_type, backbone_conf_file)
backbone = backbone_factory.get_backbone()
input = torch.randn(1, 3, 112, 112)
macs, params = profile(backbone, inputs=(input, ))
macs, params = clever_format([macs, params], "%.2f")
print('backbone type: ', backbone_type)
print('Params: ', params)
print('Macs: ', macs)

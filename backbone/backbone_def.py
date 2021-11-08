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
from backbone.EfficientNets import EfficientNet
from backbone.EfficientNets import efficientnet
from backbone.HRNet import HighResolutionNet
from backbone.GhostNet import GhostNet
from backbone.AttentionNets import ResidualAttentionNet
from backbone.TF_NAS import TF_NAS_A
from backbone.resnest.resnest import ResNeSt
from backbone.ReXNets import ReXNetV1
from backbone.LightCNN import LightCNN
from backbone.RepVGG import RepVGG
from backbone.Swin_Transformer import SwinTransformer

class BackboneFactory:
    """Factory to produce backbone according the backbone_conf.yaml.
    
    Attributes:
        backbone_type(str): which backbone will produce.
        backbone_param(dict):  parsed params and it's value. 
    """
    def __init__(self, backbone_type, backbone_conf_file):
        self.backbone_type = backbone_type
        with open(backbone_conf_file) as f:
            backbone_conf = yaml.load(f, Loader=yaml.FullLoader)
            self.backbone_param = backbone_conf[backbone_type]
        print('backbone param:')
        print(self.backbone_param)

    def get_backbone(self):
        if self.backbone_type == 'MobileFaceNet':
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            backbone = MobileFaceNet(feat_dim, out_h, out_w)
        elif self.backbone_type == 'ResNet':
            depth = self.backbone_param['depth'] # depth of the ResNet, e.g. 50, 100, 152.
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            net_mode = self.backbone_param['net_mode'] # 'ir' for improved by resnt, 'ir_se' for SE-ResNet.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            backbone = Resnet(depth, drop_ratio, net_mode, feat_dim, out_h, out_w)
        elif self.backbone_type == 'EfficientNet':
            width = self.backbone_param['width'] # width for EfficientNet, e.g. 1.0, 1.2, 1.4, ...
            depth = self.backbone_param['depth'] # depth for EfficientNet, e.g. 1.0, 1.2, 1.4, ...
            image_size = self.backbone_param['image_size'] # input image size, e.g. 112.
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            blocks_args, global_params = efficientnet(
                width_coefficient=width, depth_coefficient=depth, 
                dropout_rate=drop_ratio, image_size=image_size)
            backbone = EfficientNet(out_h, out_w, feat_dim, blocks_args, global_params)
        elif self.backbone_type == 'HRNet':
            config = {}
            config['MODEL'] = self.backbone_param
            backbone = HighResolutionNet(config)
        elif self.backbone_type == 'GhostNet':
            width = self.backbone_param['width']
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final feature
            backbone = GhostNet(width, drop_ratio, feat_dim, out_h, out_w)
        elif self.backbone_type == 'AttentionNet':
            stage1_modules = self.backbone_param['stage1_modules'] # the number of attention modules in stage1.
            stage2_modules = self.backbone_param['stage2_modules'] # the number of attention modules in stage2.
            stage3_modules = self.backbone_param['stage3_modules'] # the number of attention modules in stage3.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            backbone = ResidualAttentionNet(
                stage1_modules, stage2_modules, stage3_modules,
                feat_dim, out_h, out_w)
        elif self.backbone_type == 'TF-NAS':
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            backbone = TF_NAS_A(out_h, out_w, feat_dim, drop_ratio)
        elif self.backbone_type == 'ResNeSt':
            depth = self.backbone_param['depth'] # depth of the ResNet, e.g. 50, 100, 152.
            drop_ratio = self.backbone_param['drop_ratio'] # drop out ratio.
            feat_dim = self.backbone_param['feat_dim'] # dimension of the output features, e.g. 512.
            out_h = self.backbone_param['out_h'] # height of the feature map before the final features.
            out_w = self.backbone_param['out_w'] # width of the feature map before the final features.
            backbone = ResNeSt(depth, drop_ratio, feat_dim, out_h, out_w)
        elif self.backbone_type == 'ReXNet':
            input_ch = self.backbone_param['input_ch']
            final_ch = self.backbone_param['final_ch']
            width_mult = self.backbone_param['width_mult']
            depth_mult = self.backbone_param['depth_mult']
            use_se = True if self.backbone_param['use_se']==1 else False
            se_ratio = self.backbone_param['se_ratio']
            out_h = self.backbone_param['out_h']
            out_w = self.backbone_param['out_w']
            feat_dim = self.backbone_param['feat_dim']
            dropout_ratio = self.backbone_param['dropout_ratio']
            backbone = ReXNetV1(input_ch, final_ch, width_mult, depth_mult, use_se, se_ratio,
                                out_h, out_w, feat_dim, dropout_ratio)
        elif self.backbone_type == 'LightCNN':
            depth = self.backbone_param['depth']
            out_h = self.backbone_param['out_h']
            out_w = self.backbone_param['out_w']
            feat_dim = self.backbone_param['feat_dim']            
            drop_ratio = self.backbone_param['dropout_ratio']
            backbone = LightCNN(depth, drop_ratio, out_h, out_w, feat_dim)
        elif self.backbone_type == 'RepVGG':
            blocks1 = self.backbone_param['blocks1']
            blocks2 = self.backbone_param['blocks2']
            blocks3 = self.backbone_param['blocks3']
            blocks4 = self.backbone_param['blocks4']
            width1 = self.backbone_param['width1']
            width2 = self.backbone_param['width2']
            width3 = self.backbone_param['width3']
            width4 = self.backbone_param['width4']
            out_h = self.backbone_param['out_h']
            out_w = self.backbone_param['out_w']
            feat_dim = self.backbone_param['feat_dim']            
            backbone = RepVGG([blocks1, blocks2, blocks3, blocks4], 
                              [width1, width2, width3, width4],
                              feat_dim, out_h, out_w)
        elif self.backbone_type == 'SwinTransformer':
            img_size = self.backbone_param['img_size']
            patch_size= self.backbone_param['patch_size']
            in_chans = self.backbone_param['in_chans']
            embed_dim = self.backbone_param['embed_dim']
            depths = self.backbone_param['depths']
            num_heads = self.backbone_param['num_heads']
            window_size = self.backbone_param['window_size']
            mlp_ratio = self.backbone_param['mlp_ratio']
            drop_rate = self.backbone_param['drop_rate']
            drop_path_rate = self.backbone_param['drop_path_rate']
            backbone = SwinTransformer(img_size=img_size,
                                       patch_size=patch_size,
                                       in_chans=in_chans,
                                       embed_dim=embed_dim,
                                       depths=depths,
                                       num_heads=num_heads,
                                       window_size=window_size,
                                       mlp_ratio=mlp_ratio,
                                       qkv_bias=True,
                                       qk_scale=None,
                                       drop_rate=drop_rate,
                                       drop_path_rate=drop_path_rate,
                                       ape=False,
                                       patch_norm=True,
                                       use_checkpoint=False)
        else:
            pass
        return backbone

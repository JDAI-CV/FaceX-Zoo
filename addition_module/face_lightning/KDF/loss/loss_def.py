""" 
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com   
""" 

import sys
import yaml
sys.path.append('../../')
from loss.snn_mimic import SNN_MIMIC
from loss.soft_target import SoftTarget
from loss.fitnet import FitNet
from loss.fsp import FSP
from loss.pkt import PKTCosSim
from loss.ft import FT

class KDLossFactory:
    """Factory to produce head according to the head_conf.yaml
    
    Attributes:
        head_type(str): which head will be produce.
        head_param(dict): parsed params and it's value.
    """
    def __init__(self, loss_type, loss_conf_file):
        self.loss_type = loss_type
        with open(loss_conf_file) as f:
            loss_conf = yaml.load(f)
            self.loss_param = loss_conf[loss_type]
        print('loss param:')
        print(self.loss_param)
    def get_kd_loss(self):
        if self.loss_type == 'SNN-MIMIC':
            loss = SNN_MIMIC()
        elif self.loss_type == 'SoftTarget':
            T = self.loss_param['T']
            loss = SoftTarget(T)
        elif self.loss_type == 'FitNet':
            loss = FitNet()
        elif self.loss_type == 'FSP':
            loss = FSP()
        elif self.loss_type == 'PKT':
            loss = PKTCosSim()
        elif self.loss_type == 'FT':
            loss = FT()
        else:
            pass
        return loss

""" 
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com   
""" 

import sys
import yaml
sys.path.append('../../')
from loss.logits import Logits
from loss.st import SoftTarget
from loss.pkt import PKTCosSim

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
        if self.loss_type == 'Logits':
            loss = Logits()
        elif self.loss_type == 'SoftTarget':
            T = self.loss_param['T']
            loss = SoftTarget(T)
        elif self.loss_type == 'PKT':
            loss = PKTCosSim()
        else:
            pass
        return loss

"""
@author: Jun Wang
@date: 20201016
@contact: jun21wangustc@gmail.com
"""

import torch

class ModelLoader:
    """Load a model by network and weights file.

    Attributes: 
        model(object): the model definition file.
    """
    def __init__(self, backbone_factory):
        self.model = backbone_factory.get_backbone()

    def load_model_default(self, model_path):
        """The default method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        self.model.load_state_dict(torch.load(model_path)['state_dict'], strict=True) 
        model = torch.nn.DataParallel(self.model).cuda()
        return model

    def load_model(self, model_path):
        """The custom method to load a model.
        
        Args:
            model_path(str): the path of the weight file.
        
        Returns:
            model(object): initialized model.
        """
        model_dict = self.model.state_dict()
        pretrained_dict = torch.load(model_path)['state_dict']
        #pretrained_dict = torch.load(model_path) 
        new_pretrained_dict = {}
        for k in model_dict:
            new_pretrained_dict[k] = pretrained_dict['backbone.'+k] # tradition training
            #new_pretrained_dict[k] = pretrained_dict['feat_net.'+k] # tradition training
            #new_pretrained_dict[k] = pretrained_dict['module.'+k]
            #new_pretrained_dict[k] = pretrained_dict['module.backbone.'+k]
            #new_pretrained_dict[k] = pretrained_dict[k] # co-mining
        model_dict.update(new_pretrained_dict)
        self.model.load_state_dict(model_dict)
        model = torch.nn.DataParallel(self.model).cuda()
        return model

"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
from abc import ABCMeta, abstractmethod
import torch

class BaseModelHandler(metaclass=ABCMeta):
    """Base class for all neural network models.
    All the model loaders need to inherit this base class, 
    and each new model needs to implement the "inference_on_image" method
    """
    def __init__(self, model, device, cfg):
        """
        Generate the model by loading the configuration file.
        #######:param cfg: Cfg Node
        """
        self.model = model
        self.model.eval()
        self.cfg = cfg
        self.device = torch.device(device)

    @abstractmethod
    def inference_on_image(self, image):
        pass

    def _preprocess(self, image):
        pass

    def _postprocess(self, output):
        pass

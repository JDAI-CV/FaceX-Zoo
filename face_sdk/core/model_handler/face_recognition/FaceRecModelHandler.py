"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk')

import numpy as np
import torch

from core.model_handler.BaseModelHandler import BaseModelHandler
from utils.BuzException import *

class FaceRecModelHandler(BaseModelHandler):
    """Implementation of face recognition model handler

    Attributes:
        model: the face recognition model.
        device: use cpu or gpu to process.
        cfg(dict): testing config, inherit from the parent class.
    """
    def __init__(self, model, device, cfg):
        """
        Init FaceRecModelHandler settings. 
        """
        super().__init__(model, device, cfg)
        self.mean = self.cfg['mean']
        self.std = self.cfg['std']
        self.input_height = self.cfg['input_height']
        self.input_width = self.cfg['input_width']
        
    def inference_on_image(self, image):
        """Get the inference of the image.

        Returns:
            A numpy array, the output feature, shape (512,), 
        """
        try:
            image = self._preprocess(image)
        except Exception as e:
            raise e
        image = torch.unsqueeze(image, 0)
        image = image.to(self.device)
        with torch.no_grad():
            feature = self.model(image).cpu().numpy()
        feature = np.squeeze(feature)
        return feature

    def _preprocess(self, image):
        """Preprocess the input image.

        Returns:
           A torch tensor, the input after preprecess, shape: (3, 112, 112).
        """       
        if not isinstance(image, np.ndarray):
            logger.error('The input should be the ndarray read by cv2!')
            raise InputError()
        height, width, channels = image.shape
        if height != self.input_height or width != self.input_width:
            raise FalseImageSizeError()
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        if image.ndim == 4:
            image = image[:,:,:3]
        if image.ndim > 4:
            raise FaseChannelError(image.ndim)
        image = (image.transpose((2, 0, 1)) - self.mean) / self.std
        image = image.astype(np.float32)
        image = torch.from_numpy(image)
        return image

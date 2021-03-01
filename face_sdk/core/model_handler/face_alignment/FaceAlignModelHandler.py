"""
@author: JiXuan Xu, Jun Wang
@date: 20201023
@contact: jun21wangustc@gmail.com 
"""
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk')

import cv2
import torch
import numpy as np
import torch.backends.cudnn as cudnn

from core.model_handler.BaseModelHandler import BaseModelHandler
from utils.BuzException import *
from torchvision import transforms

class FaceAlignModelHandler(BaseModelHandler):
    """Implimentation of face landmark model handler

    Attributes:
        model: the face landmark model.
        device: use cpu or gpu to process.
        cfg(dict): testing config, inherit from the parent class.
    """
    def __init__(self, model, device, cfg):
        """
        Init FaceLmsModelHandler settings. 
        """
        super().__init__(model, device, cfg)
        self.img_size = self.cfg['img_size']
        
    def inference_on_image(self, image, dets):
        """Get the inference of the image and process the inference result.

        Returns:
            A numpy array, the landmarks prediction based on the shape of original image, shape: (106, 2), 
        """
        cudnn.benchmark = True
        try:
            image_pre = self._preprocess(image, dets)
        except Exception as e:
            raise e
        self.model = self.model.to(self.device)
        image_pre = image_pre.unsqueeze(0)
        with torch.no_grad():
            image_pre = image_pre.to(self.device)
            _, landmarks_normal = self.model(image_pre)
        landmarks = self._postprocess(landmarks_normal)
        return landmarks

    def _preprocess(self, image, det):
        """Preprocess the input image, cutting the input image through the face detection information.
        Using the face detection result(dets) to get the face position in the input image.
        After determining the center of face position and the box size of face, crop the image
        and resize it into preset size.

        Returns:
           A torch tensor, the image after preprecess, shape: (3, 112, 112).
        """
        if not isinstance(image, np.ndarray):
            logger.error('The input should be the ndarray read by cv2!')
            raise InputError()
        img = image.copy()
        self.image_org = image.copy()
        img = np.float32(img)

        xy_temp = np.array([det[0], det[1]])
        zz_temp = np.array([det[2], det[3]])
        wh = zz_temp - xy_temp + 1
        center_point = (xy_temp + wh / 2).astype(np.int32)
        boxsize_large = int(np.max(wh) * 1.2)
        xy = center_point - boxsize_large // 2
        self.xy = xy
        self.boxsize = boxsize_large
        x1, y1 = xy
        x2, y2 = xy + boxsize_large
        h, w, _ = img.shape
        dx = max(0, -x1)
        dy = max(0, -y1)
        x1 = max(0, x1)
        y1 = max(0, y1)
        edx = max(0, x2 - w)
        edy = max(0, y2 - h)
        x2 = min(w, x2)
        y2 = min(h, y2)
        image_target = image[y1:y2, x1:x2]
        if dx > 0 or dy > 0 or edx > 0 or edy > 0:
            image_target = cv2.copyMakeBorder(
                image_target, dy, edy, dx, edx, cv2.BORDER_CONSTANT, 0)

        image_target = cv2.resize(image_target, (self.img_size, self.img_size))
        t = transforms.Compose([transforms.ToTensor()])
        img_after = t(image_target)
        return img_after

    def _postprocess(self, landmarks_normal):
        """Process the predicted landmarks into the form of the original image.

        Returns:
            A numpy array, the landmarks based on the shape of original image, shape: (106, 2), 
        """    
        landmarks_normal = landmarks_normal.cpu().numpy()
        landmarks_normal = landmarks_normal.reshape(landmarks_normal.shape[0], -1, 2)
        landmarks = landmarks_normal[0] * [self.boxsize, self.boxsize] + self.xy
        return landmarks

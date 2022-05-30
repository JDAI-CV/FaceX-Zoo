"""
@author: JiXuan Xu, Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com 
"""

import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk')

import torch
import numpy as np
from math import ceil
from itertools import product as product
import torch.backends.cudnn as cudnn

from core.model_handler.BaseModelHandler import BaseModelHandler
from utils.BuzException import *


class FaceDetModelHandler(BaseModelHandler):
    """Implementation of face detection model handler

    Attributes:
        model: the face detection model.
        device: use cpu or gpu to process.
        cfg(dict): testing config, inherit from the parent class.
    """
    def __init__(self, model, device, cfg):
        """
        Init FaceDetModelHandler settings. 
        """
        super().__init__(model, device, cfg)
        self.variance = self.cfg['variance']
        
    def inference_on_image(self, image):
        """Get the inference of the image and process the inference result.

        Returns:
            A numpy array, the shape is N * (x, y, w, h, confidence), 
            N is the number of detection box.
        """
        cudnn.benchmark = True
        input_height, input_width, _ = image.shape
        try:
            image, scale = self._preprocess(image)
        except Exception as e:
            raise e
        self.model = self.model.to(self.device)
        image = torch.from_numpy(image).unsqueeze(0)
        with torch.no_grad():
            image = image.to(self.device)
            scale = scale.to(self.device)
            loc, conf, landms = self.model(image)
        dets = self._postprocess(loc, conf, scale, input_height, input_width)
        return dets

    def _preprocess(self, image):
        """Preprocess the image, such as standardization and other operations.

        Returns:
            A numpy array list, the shape is channel * h * w.
            A tensor, the shape is 4.
        """
        if not isinstance(image, np.ndarray):
            logger.error('The input should be the ndarray read by cv2!')
            raise InputError()
        img = np.float32(image)
        scale = torch.Tensor([img.shape[1], img.shape[0], img.shape[1], img.shape[0]])
        img -= (104, 117, 123)
        img = img.transpose(2, 0, 1)
        return img, scale

    def _postprocess(self, loc, conf, scale, input_height, input_width):
        """Postprecess the prediction result.
        Decode detection result, set the confidence threshold and do the NMS
        to keep the appropriate detection box. 

        Returns:
            A numpy array, the shape is N * (x, y, w, h, confidence), 
            N is the number of detection box.
        """
        priorbox = PriorBox(self.cfg, image_size=(input_height, input_width))
        priors = priorbox.forward()
        priors = priors.to(self.device)
        prior_data = priors.data
        boxes = self.decode(loc.data.squeeze(0), prior_data, self.cfg['variance'])
        boxes = boxes * scale
        boxes = boxes.cpu().numpy()
        scores = conf.squeeze(0).data.cpu().numpy()[:, 1]

        # ignore low scores
        inds = np.where(scores > self.cfg['confidence_threshold'])[0]
        boxes = boxes[inds]
        scores = scores[inds]

        # keep top-K before NMS
        order = scores.argsort()[::-1]
        boxes = boxes[order]
        scores = scores[order]

        # do NMS
        nms_threshold = 0.2
        dets = np.hstack((boxes, scores[:, np.newaxis])).astype(np.float32, copy=False)
        keep = self.py_cpu_nms(dets, nms_threshold)
        dets = dets[keep, :]
        return dets

    # Adapted from https://github.com/chainer/chainercv
    def decode(self, loc, priors, variances):
        """Decode locations from predictions using priors to undo
        the encoding we did for offset regression at train time.
        Args:
            loc (tensor): location predictions for loc layers,
                Shape: [num_priors,4]
            priors (tensor): Prior boxes in center-offset form.
                Shape: [num_priors,4].
            variances: (list[float]) Variances of priorboxes

        Return:
            decoded bounding box predictions
        """
        boxes = torch.cat((priors[:, :2], priors[:, 2:]), 1)
        boxes[:, :2] = priors[:, :2] + loc[:, :2] * variances[0] * priors[:, 2:]
        boxes[:, 2:] = priors[:, 2:] * torch.exp(loc[:, 2:] * variances[1])
        boxes[:, :2] -= boxes[:, 2:] / 2
        boxes[:, 2:] += boxes[:, :2]
        return boxes

    # Adapted from https://github.com/biubug6/Pytorch_Retinaface
    def py_cpu_nms(self, dets, thresh):
        """Python version NMS.

        Returns:
            The kept index after NMS.
        """
        x1 = dets[:, 0]
        y1 = dets[:, 1]
        x2 = dets[:, 2]
        y2 = dets[:, 3]
        scores = dets[:, 4]
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        order = scores.argsort()[::-1]
        keep = []
        while order.size > 0:
            i = order[0]
            keep.append(i)
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(0.0, xx2 - xx1 + 1)
            h = np.maximum(0.0, yy2 - yy1 + 1)
            inter = w * h
            ovr = inter / (areas[i] + areas[order[1:]] - inter)
            inds = np.where(ovr <= thresh)[0]
            order = order[inds + 1]
        return keep

# Adapted from https://github.com/biubug6/Pytorch_Retinafacey
class PriorBox(object):
    """Compute the suitable parameters of anchors for later decode operation

    Attributes:
        cfg(dict): testing config.
        image_size(tuple): the input image size.
    """
    def __init__(self, cfg, image_size=None):
        """
        Init priorBox settings related to the generation of anchors. 
        """
        super(PriorBox, self).__init__()
        self.min_sizes = cfg['min_sizes']
        self.steps = cfg['steps']
        self.image_size = image_size
        self.feature_maps = [[ceil(self.image_size[0]/step), ceil(self.image_size[1]/step)] for step in self.steps]
        self.name = "s"

    def forward(self):
        anchors = []
        for k, f in enumerate(self.feature_maps):
            min_sizes = self.min_sizes[k]
            for i, j in product(range(f[0]), range(f[1])):
                for min_size in min_sizes:
                    s_kx = min_size / self.image_size[1]
                    s_ky = min_size / self.image_size[0]
                    dense_cx = [x * self.steps[k] / self.image_size[1] for x in [j + 0.5]]
                    dense_cy = [y * self.steps[k] / self.image_size[0] for y in [i + 0.5]]
                    for cy, cx in product(dense_cy, dense_cx):
                        anchors += [cx, cy, s_kx, s_ky]
        # back to torch land
        output = torch.Tensor(anchors).view(-1, 4)
        return output

from typing import Optional, Dict, Any
import functools

# from ..util import download_jit

import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk')

import torch
import torch.nn.functional as F
import numpy as np
from math import ceil
from itertools import product as product
import torch.backends.cudnn as cudnn

from core.model_handler.BaseModelHandler import BaseModelHandler
from utils.transform import *

pretrain_settings = {
    'lapa/448': {
        'matrix_src_tag': 'points',
        'get_matrix_fn': functools.partial(get_face_align_matrix,
                                           target_shape=(448, 448), target_face_scale=1.0),
        'get_grid_fn': functools.partial(make_tanh_warp_grid,
                                         warp_factor=0.8, warped_shape=(448, 448)),
        'get_inv_grid_fn': functools.partial(make_inverted_tanh_warp_grid,
                                             warp_factor=0.8, warped_shape=(448, 448)),
        'label_names': ['background', 'face', 'rb', 'lb', 're',
                        'le', 'nose',  'ulip', 'imouth', 'llip', 'hair']
    }
}


class FaceParsingModelHandler(BaseModelHandler):
    """ The face parsing models from [FaRL](https://github.com/FacePerceiver/FaRL).
    Please consider citing 
    ```bibtex
        @article{zheng2021farl,
            title={General Facial Representation Learning in a Visual-Linguistic Manner},
            author={Zheng, Yinglin and Yang, Hao and Zhang, Ting and Bao, Jianmin and Chen, 
                Dongdong and Huang, Yangyu and Yuan, Lu and Chen, 
                Dong and Zeng, Ming and Wen, Fang},
            journal={arXiv preprint arXiv:2112.03109},
            year={2021}
        }
    ```
    """
    def __init__(self, model=None, device=None, cfg=None):
        super().__init__(model, device, cfg)
        
#         self.conf_name = conf_name
        self.model = model.to(self.device)
#         self.eval()
    def _preprocess(self, image, face_nums):
        """Preprocess the image, such as standardization and other operations.

        Returns:
            A tensor, the shape is 1 x 3 x h x w.
            A dict, {'rects','points','scores','image_ids'} 
        """
        if not isinstance(image, np.ndarray):
            logger.error('The input should be the ndarray read by cv2!')
            raise InputError()
        img = np.float32(image)
        img = img.transpose(2, 0, 1)
        img = np.expand_dims(img,0).repeat(face_nums,axis=0)
        return torch.from_numpy(img)
    def inference_on_image(self, face_nums: int, images: torch.Tensor, landmarks):
        """Get the inference of the image and process the inference result.

        Returns:
             
        """
        cudnn.benchmark = True
        try:
            image_pre = self._preprocess(images, face_nums)
        except Exception as e:
            raise e
        setting = pretrain_settings['lapa/448']
        images = image_pre.float() / 255.0
        _, _, h, w = images.shape
        simages = images.to(self.device)#data_pre['image_ids']
        matrix = setting['get_matrix_fn'](landmarks.to(self.device))
        grid = setting['get_grid_fn'](matrix=matrix, orig_shape=(h, w))
        inv_grid = setting['get_inv_grid_fn'](matrix=matrix, orig_shape=(h, w))

        w_images = F.grid_sample(
            simages, grid, mode='bilinear', align_corners=False)

        w_seg_logits, _ = self.model(w_images)  # (b*n) x c x h x w

        seg_logits = F.grid_sample(
            w_seg_logits, inv_grid, mode='bilinear', align_corners=False)
        data_pre = {}
        data_pre['seg'] = {'logits': seg_logits,
                       'label_names': setting['label_names']}
        return data_pre
    
    def _postprocess(self, loc, conf, scale, input_height, input_width):
        """Postprecess the prediction result.
        Decode detection result, set the confidence threshold and do the NMS
        to keep the appropriate detection box. 

        Returns:
            A numpy array, the shape is N * (x, y, w, h, confidence), 
            N is the number of detection box.
        """
        pass
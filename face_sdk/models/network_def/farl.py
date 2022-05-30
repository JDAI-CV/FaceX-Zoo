from typing import Optional, Dict, Any
import functools
import torch
import torch.nn.functional as F

# from ..util import download_jit
from ..transform import (get_crop_and_resize_matrix, get_face_align_matrix,
                         make_inverted_tanh_warp_grid, make_tanh_warp_grid)
from .base import FaceParser

pretrain_settings = {
    'lapa/448': {
        'url': [
            'https://github.com/FacePerceiver/facer/releases/download/models-v1/face_parsing.farl.lapa.main_ema_136500_jit191.pt',
        ],
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


class FaRLFaceParser(FaceParser):
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

    def __init__(self, conf_name: Optional[str] = None,
                 model_path: Optional[str] = None, device=None) -> None:
        super().__init__()
        if conf_name is None:
            conf_name = 'lapa/448'
        if model_path is None:
            model_path = pretrain_settings[conf_name]['url']
        self.conf_name = conf_name
        self.net = download_jit(model_path, map_location=device)
        self.eval()

    def forward(self, images: torch.Tensor, data: Dict[str, Any]):
        setting = pretrain_settings[self.conf_name]
        images = images.float() / 255.0
        _, _, h, w = images.shape

        simages = images[data['image_ids']]
        matrix = setting['get_matrix_fn'](data[setting['matrix_src_tag']])
        grid = setting['get_grid_fn'](matrix=matrix, orig_shape=(h, w))
        inv_grid = setting['get_inv_grid_fn'](matrix=matrix, orig_shape=(h, w))

        w_images = F.grid_sample(
            simages, grid, mode='bilinear', align_corners=False)

        w_seg_logits, _ = self.net(w_images)  # (b*n) x c x h x w

        seg_logits = F.grid_sample(
            w_seg_logits, inv_grid, mode='bilinear', align_corners=False)

        data['seg'] = {'logits': seg_logits,
                       'label_names': setting['label_names']}
        return data
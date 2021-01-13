"""
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com 
"""

import os
import numpy as np
from math import sqrt

l2_norm = lambda feat: feat / sqrt(feat.dot(feat))
def concat_feat(feat1, feat2):
    #feat1 = l2_norm(feat1)
    #feat2 = l2_norm(feat2)
    feat = np.hstack([feat1, feat2])
    return feat

def concat_save(feat_m_dir, feat_e_dir, feat_dir_new):
    for root, dirs, files in os.walk(feat_m_dir):
        for cur_file in files:
            cur_mask_feat_path = os.path.join(root, cur_file)
            assert(os.path.exists(cur_mask_feat_path))
            cur_mask_feat = np.load(cur_mask_feat_path)
            short_path = cur_mask_feat_path[len(feat_m_dir)+1:]
            cur_eye_feat_path = os.path.join(feat_e_dir, short_path)
            assert(os.path.exists(cur_eye_feat_path))
            cur_eye_feat = np.load(cur_eye_feat_path)
            cur_feat_concat = concat_feat(cur_mask_feat, cur_eye_feat)
            cur_concat_feat_path = os.path.join(feat_dir_new, short_path)
            cur_concat_feat_dir = os.path.dirname(cur_concat_feat_path)
            if not os.path.exists(cur_concat_feat_dir):
                os.makedirs(cur_concat_feat_dir)
            np.save(cur_concat_feat_path, cur_feat_concat)
            
if __name__ == '__main__':
    feat_m_dir = 'feats_mask'
    feat_e_dir = 'feats_eye'
    feat_dir_new = 'feats_concat'
    concat_save(feat_m_dir, feat_e_dir, feat_dir_new)

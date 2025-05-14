"""
@author: Jun Wang
@date: 20201016 
@contact: jun21wangustc@gmail.com 
"""

import os
import logging as logger
import numpy as np
import torch
import torch.nn.functional as F
logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class CommonExtractor:
    """Common feature extractor.
    
    Attributes:
        device(object): device to init model.
    """
    def __init__(self, device):
        self.device = torch.device(device)

    def extract_online(self, model, data_loader):
        """Extract and return features.
        
        Args:
            model(object): initialized model.
            data_loader(object): load data to be extracted.

        Returns:
            image_name2feature(dict): key is the name of image, value is feature of image.
        """
        model.eval()
        image_name2feature = {}
        with torch.no_grad(): 
            for batch_idx, (images, filenames) in enumerate(data_loader):
                images = images.to(self.device)
                features = model(images)
                features = F.normalize(features)
                features = features.cpu().numpy()
                for filename, feature in zip(filenames, features): 
                    image_name2feature[filename] = feature
        return image_name2feature

    def extract_offline(self, feats_root, model, data_loader):
        """Extract and save features.

        Args:
            feats_root(str): the path to save features.
            model(object): initialized model.
            data_loader(object): load data to be extracted.
        """
        model.eval()
        with torch.no_grad():
            for batch_idx, (images, filenames) in enumerate(data_loader):
                images = images.to(self.device)
                features = model(images)
                features = F.normalize(features)
                features = features.cpu().numpy()
                for filename, feature in zip(filenames, features):
                    feature_name = os.path.splitext(filename)[0]
                    feature_path = os.path.join(feats_root, feature_name + '.npy')
                    feature_dir = os.path.dirname(feature_path)
                    if not os.path.exists(feature_dir):
                        os.makedirs(feature_dir)
                    np.save(feature_path, feature)
                if (batch_idx + 1) % 10 == 0:
                    logger.info('Finished batches: %d/%d.' % (batch_idx+1, len(data_loader)))

class FeatureHandler:
    """Some method to deal with features.
    
    Atributes:
        feats_root(str): the directory which the fetures in.
    """
    def __init__(self, feats_root):
        self.feats_root = feats_root

    def load_feature(self):
        """Load features to memory.
        
        Returns:
            image_name2feature(dict): key is the name of image, value is feature of image.
        """
        image_name2feature = {}
        for root, dirs, files in os.walk(self.feats_root):
            for cur_file in files: 
                if cur_file.endswith('.npy'):
                    cur_file_path = os.path.join(root, cur_file)
                    cur_feats = np.load(cur_file_path)
                    if self.feats_root.endswith('/'):
                        cur_short_path = cur_file_path[len(self.feats_root) : ]
                    else:
                        cur_short_path = cur_file_path[len(self.feats_root) + 1 : ]
                    cur_key = cur_short_path.replace('.npy', '.jpg')
                    image_name2feature[cur_key] = cur_feats
        return image_name2feature

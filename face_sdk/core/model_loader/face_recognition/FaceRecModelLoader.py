"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk')

import torch

from core.model_loader.BaseModelLoader import BaseModelLoader

class FaceRecModelLoader(BaseModelLoader):
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        logger.info('Start to analyze the face recognition model, model path: %s, model category: %s，model name: %s' %
                    (model_path, model_category, model_name))
        super().__init__(model_path, model_category, model_name, meta_file)
        self.cfg['mean'] = self.meta_conf['mean']
        self.cfg['std'] = self.meta_conf['std']
        
    def load_model(self):
        try:
            model = torch.load(self.cfg['model_file_path'])
        except Exception as e:
            logger.error('The model failed to load, please check the model path: %s!'
                         % self.cfg['model_file_path'])
            raise e
        else:
            logger.info('Successfully loaded the face recognition model!')
            return model, self.cfg

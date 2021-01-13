"""
@author: JiXuan Xu, Jun Wang
@date: 20201015
@contact: jun21wangustc@gmail.com 
"""
import os
import sys
sys.path.append('models/network_def')
import logging.config
logging.config.fileConfig("config/logging.conf")
logger = logging.getLogger('sdk') 
from abc import ABCMeta, abstractmethod

import json

class BaseModelLoader(metaclass=ABCMeta):
    """Base class for all model loader.
    All the model loaders need to inherit this base class, 
    and each new model needs to implement the "load model" method
    """
    def __init__(self, model_path, model_category, model_name, meta_file='model_meta.json'):
        model_root_dir = os.path.join(model_path, model_category, model_name)
        meta_file_path = os.path.join(model_root_dir, meta_file)
        self.cfg = {}
        try:
            self.meta_conf = json.load(open(meta_file_path, 'r'))
        except IOError as e:
            logger.error('The configuration file meta.json was not found or failed to parse the file!')
            raise e
        except Exception as e:
            logger.info('The configuration file format is wrong!')
            raise e
        else:
            logger.info('Successfully parsed the model configuration file meta.json!')
        # common configs for all model
        self.cfg['model_path'] = model_path
        self.cfg['model_category'] = model_category
        self.cfg['model_name'] = model_name
        self.cfg['model_type'] = self.meta_conf['model_type']
        self.cfg['model_info'] = self.meta_conf['model_info']
        self.cfg['model_file_path'] = os.path.join(model_root_dir, self.meta_conf['model_file'])
        self.cfg['release_date'] = self.meta_conf['release_date']
        self.cfg['input_height'] = self.meta_conf['input_height']
        self.cfg['input_width'] = self.meta_conf['input_width']

    @abstractmethod
    def load_model(self):
        """Should be overridden by all subclasses.
        Different models may have different configuration information,
        such as mean, so each model implements its own loader
        """
        pass

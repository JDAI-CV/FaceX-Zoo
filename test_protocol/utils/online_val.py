"""
@author Jun Wang
@data 20210630
@contact: jun21wangustc@gmail.com
"""

import sys
import yaml
import logging as logger
from torch.utils.data import DataLoader

sys.path.append('../../')
from data_processor.test_dataset import CommonTestDataset
from test_protocol.utils.extractor.feature_extractor import CommonExtractor
from test_protocol.lfw.lfw_evaluator import LFWEvaluator 
from test_protocol.lfw.pairs_parser import PairsParserFactory
from test_protocol.ijbc.ijbc_evaluator import IJBCEvaluator 

logger.basicConfig(level=logger.INFO,
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

class Evaluator:
    """Evaluator the trained model online.

    Attributes:
        test_set(str): which test set to use, currently surpport LFW, CPLFW, ..., IJB-C
        ijbc_evaluator(obj): evaluate model on ijbc 1:1 protocol.
        lfw_evaluator(obj): evaluate model on LFW, CPLFW, ...
    """
    def __init__(self, test_set, data_conf_file):
        self.test_set = test_set
        feature_extractor = CommonExtractor('cuda:0')
        with open(data_conf_file) as f:
            data_conf = yaml.load(f)[test_set]
            if test_set == 'IJBC':
                cropped_face_folder = data_conf['cropped_face_folder']
                image_list_file_path = data_conf['image_list_file_path']
                template_media_list_path = data_conf['template_media_list_path']
                template_pair_list_path = data_conf['template_pair_list_path']
                image_score_list_path = data_conf['image_score_list_path']
                ijbc_data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False),
                                              batch_size=5120, num_workers=4, shuffle=False)
                self.ijbc_evaluator = IJBCEvaluator(template_media_list_path, template_pair_list_path, 
                                                    image_score_list_path, ijbc_data_loader, feature_extractor)
            else:
                pairs_file_path = data_conf['pairs_file_path']
                cropped_face_folder = data_conf['cropped_face_folder']
                image_list_file_path = data_conf['image_list_file_path']
                lfw_data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False),
                                             batch_size=512, num_workers=4, shuffle=False)
                pairs_parser_factory = PairsParserFactory(pairs_file_path, test_set)
                self.lfw_evaluator = LFWEvaluator(lfw_data_loader, pairs_parser_factory, feature_extractor)

    def evaluate(self, model):
        """ Method to evaluate the input model.

        Args:
            model(obj): the loaded model.
        """
        if self.test_set == 'IJBC':
            tpr_list = self.ijbc_evaluator.test(model)
            logger.info('tpr@1e-6: %f, tpr@1e-5: %f, tpr@1e-4: %f, tpr@1e-3: %f, tpr@1e-2: %f, tpr@1e-1: %f.' % 
                        (tpr_list[5], tpr_list[4], tpr_list[3], tpr_list[2], tpr_list[1], tpr_list[0]))
        else:
            mean, std = self.lfw_evaluator.test(model)
            logger.info('accuracy on %s, mean: %f, std: %f.' % (self.test_set, mean, std))
        model.train()

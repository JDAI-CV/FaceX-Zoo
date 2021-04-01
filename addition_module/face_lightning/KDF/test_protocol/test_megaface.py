"""
@author: Jun Wang
@date: 20201012 
@contact: jun21wangustc@gmail.com
"""  

import argparse
import yaml
from megaface.megaface_evaluator import CommonMegaFaceEvaluator
from megaface.megaface_evaluator import MaskedMegaFaceEvaluator

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='megaface test protocal in python.')
    conf.add_argument("--data_conf_file", type = str, 
                      help = "The path of data_conf.yaml.")
    conf.add_argument("--max_rank", type = int, 
                      help = "Rank N accuray..")
    conf.add_argument("--facescrub_feature_dir", type = str, 
                      help = "The dir of facescrub features.")
    conf.add_argument("--megaface_feature_dir", type = str, 
                      help = "The dir of megaface features.")
    conf.add_argument("--masked_facescrub_feature_dir", type = str, 
                      help = "The dir of masked facescrub features.")
    conf.add_argument("--is_concat", type = int, 
                      help = "If the feature is concated by two nomalized features.")
    args = conf.parse_args()
    with open(args.data_conf_file) as f:
        data_conf  = yaml.load(f)['MegaFace']
        facescrub_json_list = data_conf['facescrub_list']
        megaface_json_list = data_conf['megaceface_list']
        megaface_mask = data_conf['megaface-mask']
    is_concat = True if args.is_concat == 1 else False
    if megaface_mask == 0:
        megaFaceEvaluator = CommonMegaFaceEvaluator(
            facescrub_json_list, megaface_json_list, 
            args.facescrub_feature_dir, args.megaface_feature_dir, 
            is_concat)
    elif megaface_mask == 1:
        megaFaceEvaluator = MaskedMegaFaceEvaluator(
            facescrub_json_list, megaface_json_list, 
            args.facescrub_feature_dir, args.megaface_feature_dir, 
            args.masked_facescrub_feature_dir, is_concat)
    megaFaceEvaluator.test_cmc(args.max_rank)

""" 
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""

import os
import sys
import argparse
import yaml
from prettytable import PrettyTable
from torch.utils.data import DataLoader
from ijbc.ijbc_evaluator import IJBCEvaluator
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('..')
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory

def accu_key(elem):
    return elem[1]

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='lfw test protocal.')
    conf.add_argument("--data_conf_file", type = str, 
                      help = "the path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type = str, 
                      help = "Resnet, Mobilefacenets..")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help = "The path of backbone_conf.yaml.")
    conf.add_argument('--batch_size', type = int, default = 1024)
    conf.add_argument('--model_path', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of model or the directory which some models in.')
    args = conf.parse_args()
    # parse config.
    with open(args.data_conf_file) as f:
        data_conf = yaml.load(f)['IJBC']
        cropped_face_folder = data_conf['cropped_face_folder']
        image_list_file_path = data_conf['image_list_file_path']
        template_media_list_path = data_conf['template_media_list_path']
        template_pair_list_path = data_conf['template_pair_list_path']
        image_score_list_path = data_conf['image_score_list_path']
        
    # define dataloader
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file_path, False), 
                             batch_size=args.batch_size, num_workers=4, shuffle=False)
    #model def
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    model_loader = ModelLoader(backbone_factory)
    feature_extractor = CommonExtractor('cuda:0')

    ijbc_evaluator = IJBCEvaluator(template_media_list_path, template_pair_list_path, 
                                   image_score_list_path, data_loader, feature_extractor)
    pretty_tabel = PrettyTable(["model_name", "1e-6", "1e-5", "1e-4", "1e-3", "1e-2", "1e-1"])
    accu_list = []
    if os.path.isdir(args.model_path):
        model_name_list = os.listdir(args.model_path)
        for model_name in model_name_list:
            if model_name.endswith('.pt'):
                model_path = os.path.join(args.model_path, model_name)
                model = model_loader.load_model(model_path)
                tpr_list = ijbc_evaluator.test(model)
                cur_accu = []
                cur_accu.append(os.path.basename(model_path))
                cur_accu.extend(tpr_list)
                accu_list.append(cur_accu)
        accu_list.sort(key = accu_key, reverse=True)
    else:
        model = model_loader.load_model(args.model_path)
        tpr_list = ijbc_evaluator.test(model)
        cur_accu = []
        cur_accu.append(os.path.basename(args.model_path))
        cur_accu.extend(tpr_list)
        accu_list.append(cur_accu)

    for accu_item in accu_list:
        pretty_tabel.add_row(accu_item)
    print(pretty_tabel)

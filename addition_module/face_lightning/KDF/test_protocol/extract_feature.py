""" 
@author: Jun Wang
@date: 20201012
@contact: jun21wangustc@gmail.com
"""  

import sys
import yaml
import argparse
import torch
from torch.utils.data import Dataset, DataLoader
from utils.model_loader import ModelLoader
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('..')
from data_processor.test_dataset import CommonTestDataset
from backbone.backbone_def import BackboneFactory

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='extract features for megaface.')
    conf.add_argument("--data_conf_file", type = str, 
                      help = "The path of data_conf.yaml.")
    conf.add_argument("--backbone_type", type = str, 
                      help = "Resnet, Mobilefacenets.")
    conf.add_argument("--backbone_conf_file", type = str, 
                      help = "The path of backbone_conf.yaml.")
    conf.add_argument('--batch_size', type = int, default = 1024)
    conf.add_argument('--model_path', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of model')
    conf.add_argument('--feats_root', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path for feature save.')
    args = conf.parse_args()
    with open(args.data_conf_file) as f:
        data_conf = yaml.load(f)['MegaFace']
        cropped_face_folder = data_conf['cropped_face_folder']
        image_list_file = data_conf['image_list_file']
        megaface_mask = data_conf['megaface-mask']
        masked_cropped_face_folder = data_conf['masked_cropped_face_folder']
        masked_image_list_file = data_conf['masked_image_list_file']
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file, False), 
                             batch_size=args.batch_size, num_workers=4, shuffle=False)
    # define model.
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)
    model_loader = ModelLoader(backbone_factory)
    model = model_loader.load_model(args.model_path)
    # extract feature.
    feature_extractor = CommonExtractor('cuda:0')
    feature_extractor.extract_offline(args.feats_root, model, data_loader)
    if megaface_mask == 1:
        data_loader = DataLoader(CommonTestDataset(masked_cropped_face_folder, masked_image_list_file, False), 
                                 batch_size=args.batch_size, num_workers=4, shuffle=False)
        feature_extractor.extract_offline(args.feats_root, model, data_loader)

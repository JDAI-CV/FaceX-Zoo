"""
@author: Jun Wang
@date: 20201016 
@contact: jun21wangustc@gmail.com
"""

import sys
import torch
from torch.utils.data import Dataset, DataLoader
sys.path.append('/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/data_processor')
from test_dataset import CommonTestDataset
sys.path.append('/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/test_protocol')
from utils.extractor.feature_extractor import CommonExtractor
sys.path.append('/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/backbone')
from mobilefacenet_def import MobileFaceNet

if __name__ == '__main__':
    cropped_face_folder = '/export2/wangjun492/face_database/public_data/final_data/test_data/megaface/face_crop_arcface'
    image_list_file = '/export2/wangjun492/face_database/public_data/final_data/test_data/megaface/face_crop_arcface/img_list.txt'

    #cropped_face_folder = '/export2/wangjun492/face_database/public_data/final_data/test_data/megaface/facescrub_mask_crop_arcface'
    #image_list_file = '/export2/wangjun492/face_database/public_data/meta_data/test_data/megaface/mid_files/facescrub_img_list.txt'

    model_path = '/export/home/wangjun492/wj_armory/faceX-Zoo/face_recognition/trained_models/traditional_training/Mobilefacenets/Epoch_13.pt'
    feats_root = 'feat_dir'

    # extract feat by model.
    #data_loader = DataLoader(CommonTestDataset_eye(cropped_face_folder, image_list_file), batch_size=1024, num_workers=4, shuffle=False)
    data_loader = DataLoader(CommonTestDataset(cropped_face_folder, image_list_file), batch_size=10240, num_workers=4, shuffle=False)
    model = MobileFaceNet(512, 7, 7)
    model_dict = model.state_dict()
    pretrained_dict = torch.load(model_path)['state_dict']
    new_pretrained_dict = {}
    for k in model_dict:
        new_pretrained_dict[k] = pretrained_dict['feat_net.'+k]
    model_dict.update(new_pretrained_dict)
    model.load_state_dict(model_dict)
    model = torch.nn.DataParallel(model).cuda()
    extractor = CommonExtractor('cuda:0')
    extractor.extract_offline(feats_root, model, data_loader)

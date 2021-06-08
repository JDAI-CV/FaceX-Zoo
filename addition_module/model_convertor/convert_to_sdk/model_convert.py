"""
@author: Jun Wang 
@date: 20210207
@contact: jun21wangustc@gmail.com
"""
import sys
import torch
sys.path.append('.')

from models.network_def.mobilefacenet_def import MobileFaceNet

model = MobileFaceNet(512, 7, 7)
model_dict = model.state_dict()

pretrained_model = '/export2/wangjun492/face_database/facex-zoo/share_file/trained_models/conventional_training/multi_backbone/MobileFaceNet/Epoch_17.pt'
pretrained_dict = torch.load(pretrained_model)['state_dict']

new_pretrained_dict = {}
for k in model_dict:
    new_pretrained_dict[k] = pretrained_dict['backbone.'+k]

model_dict.update(new_pretrained_dict)
model.load_state_dict(model_dict)
model.cuda()
torch.save(model, 'face_recognition_mv.pkl')

import os
from PIL import Image

import torch
import torch.nn.functional as F
import torchvision.transforms as T

from models.make_target_model import make_target_model


class Config:
    pass
cfg = Config()
cfg.ori_shape = (256, 256)
cfg.image_crop_size = (224, 224)
cfg.normalize_mean = [0.5, 0.5, 0.5]
cfg.normalize_std = [0.5, 0.5, 0.5]
cfg.last_stride = 2
cfg.num_classes = 8
cfg.num_branches = cfg.num_classes + 1
cfg.backbone = 'resnet18' # 'resnet18', 'resnet50_ibn'
cfg.pretrained = "./weights/AffectNet_res18_acc0.6285.pth"
cfg.pretrained_choice = '' # '' or 'convert'
cfg.bnneck = True  
cfg.BiasInCls = False


def inference(model, img_path, transform, is_cuda=True):
    img = Image.open(img_path).convert('RGB')    
    img_tensor = transform(img).unsqueeze(0)
    if is_cuda:
        img_tensor = img_tensor.cuda()
    
    model.eval()
    if is_cuda:
        model = model.cuda()

    pred = model(img_tensor)
    prob = F.softmax(pred, dim=-1)
    idx  = torch.argmax(prob.cpu()).item()

    key = {0: 'Neutral', 1:'Happy', 2:'Sad', 3:'Surprise', 4:'Fear', 5:'Disgust', 6:'Anger', 7:'Contempt'}
    print('Predicted: {}'.format(key[idx]))
    print('Probabilities:')
    for i in range(cfg.num_classes):
        print('{} ----> {}'.format(key[i], round(prob[0,i].item(), 4)))


if __name__ == '__main__':
    img_path = './images/test1.jpg'

    transform = T.Compose([
            T.Resize(cfg.ori_shape),
            T.CenterCrop(cfg.image_crop_size),
            T.ToTensor(),
            T.Normalize(mean=cfg.normalize_mean, std=cfg.normalize_std),
        ])

    print('Building model......')
    model = make_target_model(cfg)
    model.load_param(cfg)
    print('Loaded pretrained model from {0}'.format(cfg.pretrained))

    inference(model, img_path, transform, is_cuda=True)
    
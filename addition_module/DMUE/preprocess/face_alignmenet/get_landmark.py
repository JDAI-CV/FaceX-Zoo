#coding:utf-8
import matplotlib
import math
import torch
import copy
import time
from torch.autograd import Variable
import shutil
from skimage import io
import numpy as np
from .utils.utils import fan_NME, show_landmarks, get_preds_fromhm
from PIL import Image, ImageDraw
from pylab import *
import os
import sys
import cv2
import matplotlib.pyplot as plt
import torch
import argparse
import numpy as np
import torch.nn as nn
import time
import os
from .core import models


def load_model():
    NUM_LANDMARKS = 98

    PRETRAINED_WEIGHTS = os.path.join(os.path.dirname(__file__), './ckpt/WFLW_4HG.pth')
    GRAY_SCALE = False
    HG_BLOCKS = 4
    END_RELU = False

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # print(device)

    model_ft = models.FAN(HG_BLOCKS, END_RELU, GRAY_SCALE, NUM_LANDMARKS)

    checkpoint = torch.load(PRETRAINED_WEIGHTS, map_location=torch.device('cpu'))
    pretrained_weights = checkpoint['state_dict']
    model_weights = model_ft.state_dict()
    pretrained_weights = {k: v for k, v in pretrained_weights.items() \
                          if k in model_weights}
    model_weights.update(pretrained_weights)
    model_ft.load_state_dict(model_weights, device)

    model_ft = model_ft.to(device)
    model_ft.eval()

    return model_ft


def get_lmd(model_ft, img):
    model_ft.eval()
    # Iterate over data.
    with torch.no_grad():
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_CUBIC)
        img = img.transpose(2, 0, 1)[np.newaxis, :, :, :].astype(np.float32) / 255
        '''
        这里输入的取值范围在0-1
    
        '''
        inputs = torch.from_numpy(img).cuda()
        outputs, boundary_channels = model_ft(inputs)

        pred_heatmap = outputs[-1][:, :-1, :, :][0].detach().cpu()  # （98， 64， 64）
        pred_landmarks, _ = get_preds_fromhm(pred_heatmap.unsqueeze(0))
        pred_landmarks = pred_landmarks.squeeze().numpy() # 98 x 2 list
        pred_landmarks = (pred_landmarks * 256 / 64).astype(np.int)

        return pred_landmarks


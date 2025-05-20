import os
import time
import argparse
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils

from data import *
from networks import *
from misc import *

parser = argparse.ArgumentParser()

parser.add_argument('--gpu_ids', default='0,1,2,3', type=str)
parser.add_argument('--hdim', default=128, type=int)
parser.add_argument('--attack_type', default=4, type=int, help='attack type number')
parser.add_argument('--pre_model', default='./result/model_oulu_p1/netG_model_epoch_200_iter_0.pth', type=str)
parser.add_argument('--out_path', default='./fake_images/oulu_p1/', type=str)

spoof_img_name = 'p1_0_'
live_img_name = 'p1_1_'


def main():
    global opt, model
    args = parser.parse_args()
    print(args)

    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_ids
    cudnn.benchmark = True

    if not os.path.exists(args.out_path):
        os.makedirs(args.out_path)

    netE_nir, netE_vis, netG, netCls = define_G(hdim=args.hdim, attack_type=args.attack_type)
    load_model(netG, args.pre_model)
    netG.eval()

    num = 0
    for n in range(200):
        noise = torch.zeros(100, args.hdim).normal_(0, 1)
        noise_s = torch.zeros(100, args.hdim).normal_(0, 1)
        noise = torch.cat((noise_s, noise, noise), dim=1)
        noise = noise.cuda()

        fake = netG(noise)

        spoof = fake[:, 0:3, :, :].data.cpu().numpy()
        live = fake[:, 3:6, :, :].data.cpu().numpy()

        os.makedirs(os.path.join(args.out_path, spoof_img_name + str(n)))
        os.makedirs(os.path.join(args.out_path, live_img_name + str(n)))

        for i in range(spoof.shape[0]):
            num = num + 1
            save_img = spoof[i, :, :, :]
            save_img = np.transpose((255 * save_img).astype('uint8'), (1, 2, 0))
            output = Image.fromarray(save_img)
            save_name = str(i) + '_scene.jpg'
            output.save(os.path.join(args.out_path, spoof_img_name + str(n), save_name))

            save_img = live[i, :, :, :]
            save_img = np.transpose((255 * save_img).astype('uint8'), (1, 2, 0))
            output = Image.fromarray(save_img)
            save_name = str(i) + '_scene.jpg'
            output.save(os.path.join(args.out_path, live_img_name + str(n), save_name))

        print(num)


if __name__ == "__main__":
    main()

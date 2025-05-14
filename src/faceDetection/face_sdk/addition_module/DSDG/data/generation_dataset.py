import os
import math
import random
import numpy as np
import cv2
from PIL import Image
from collections import defaultdict

import torch
import torch.utils.data as data
import torchvision.transforms as transforms
"""
domain0 => spoof
domain1 => live
"""


class GenDataset_s(data.Dataset):
    def __init__(self, img_root, list_file, attack_type):
        super(GenDataset_s, self).__init__()

        self.img_root = img_root
        self.list_file = list_file
        self.attack_type = attack_type

        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

        self.img_spoof_list, self.make_pair_dict = self.file_reader()

    def __getitem__(self, index):
        img_line = self.img_spoof_list[index]
        img_name_spoof, label, domain_flag, spoof_label = img_line.strip().split(' ')
        if int(spoof_label) == 2:
            spoof_type = 0
        elif int(spoof_label) == 3:
            spoof_type = 1
        elif int(spoof_label) == 4:
            spoof_type = 2
        elif int(spoof_label) == 5:
            spoof_type = 3
        else:
            print('spoof type out of except')
            spoof_type = -1
            exit()

        img_name_live = self.get_pair(label, '1')

        img_domain0 = cv2.imread(os.path.join(self.img_root, 'train_img_flod/', img_name_spoof))
        img_domain0 = cv2.cvtColor(img_domain0, cv2.COLOR_BGR2RGB)
        img_domain1 = cv2.imread(os.path.join(self.img_root, 'train_img_flod/', img_name_live))
        img_domain1 = cv2.cvtColor(img_domain1, cv2.COLOR_BGR2RGB)

        bbox_domain0_path = os.path.join(self.img_root, 'train_bbox_flod/', img_name_spoof[:-4] + '.dat')
        bbox_domain1_path = os.path.join(self.img_root, 'train_bbox_flod/', img_name_live[:-4] + '.dat')

        face_scale = 1.0

        img_domain0_crop = cv2.resize(self.crop_face_from_scene(img_domain0, bbox_domain0_path, face_scale), (256, 256))
        img_domain1_crop = cv2.resize(self.crop_face_from_scene(img_domain1, bbox_domain1_path, face_scale), (256, 256))

        img_domain0 = self.transform(img_domain0_crop)
        img_domain1 = self.transform(img_domain1_crop)

        return {'0': img_domain0, '1': img_domain1, 'type': spoof_type}

    def __len__(self):
        return len(self.img_spoof_list)

    def file_reader(self):

        def dict_profile():
            return {'0': [], '1': [], 'type': []}

        with open(self.list_file) as file:
            img_list = file.readlines()
            img_list = [x.strip() for x in img_list]

        make_pair_dict = defaultdict(dict_profile)
        img_spoof_list = []

        for line in img_list:
            img_name, label, domain_flag, spoof_label = line.strip().split(' ')
            make_pair_dict[label][domain_flag].append(img_name)

            if domain_flag == '0':
                img_spoof_list.append(line)

        return img_spoof_list, make_pair_dict

    def get_pair(self, label, domain_flag):
        img_name = random.choice(self.make_pair_dict[label][domain_flag])
        return img_name

    def crop_face_from_scene(self, image, face_name_full, scale):
        f = open(face_name_full, 'r')
        lines = f.readlines()
        lines = lines[0].split(' ')
        y1, x1, w, h = [int(ele) for ele in lines[:4]]
        f.close()
        y2 = y1 + w
        x2 = x1 + h

        y_mid = (y1 + y2) / 2.0
        x_mid = (x1 + x2) / 2.0
        h_img, w_img = image.shape[0], image.shape[1]
        # w_img,h_img=image.size
        w_scale = scale * w
        h_scale = scale * h
        y1 = y_mid - w_scale / 2.0
        x1 = x_mid - h_scale / 2.0
        y2 = y_mid + w_scale / 2.0
        x2 = x_mid + h_scale / 2.0
        y1 = max(math.floor(y1), 0)
        x1 = max(math.floor(x1), 0)
        y2 = min(math.floor(y2), w_img)
        x2 = min(math.floor(x2), h_img)

        region = image[x1:x2, y1:y2]
        return region

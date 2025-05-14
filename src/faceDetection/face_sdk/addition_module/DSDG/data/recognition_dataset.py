import os
import copy
import math
import time
import random
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
import torchvision.transforms as transforms


def default_loader(path):
    img = Image.open(path).convert('L')
    return img


def default_list_reader(fileList):
    imgList = []
    with open(fileList, 'r') as file:
        for line in file.readlines():
            imgPath, label, domain = line.strip().split(' ')
            imgList.append((imgPath, int(label), int(domain)))
    return imgList


class ImageList(data.Dataset):
    def __init__(self, root, fileList, list_reader=default_list_reader, loader=default_loader):
        self.root = root
        self.imgList = list_reader(fileList)
        self.loader = loader

        self.transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.ToTensor(),
        ])

    def __getitem__(self, index):
        imgPath, target, domain = self.imgList[index]
        img = self.loader(os.path.join(self.root, imgPath))

        img = self.transform(img)

        return {'img': img, 'label': target, 'domain_flag': domain}

    def __len__(self):
        return len(self.imgList)


class SeparateBatchSampler(object):
    def __init__(self, real_data_idx, fake_data_idx, batch_size=128, ratio=0.5, put_back=False):
        self.batch_size = batch_size
        self.ratio = ratio
        self.real_data_num = len(real_data_idx)
        self.fake_data_num = len(fake_data_idx)
        self.max_num_image = max(self.real_data_num, self.fake_data_num)

        self.real_data_idx = real_data_idx
        self.fake_data_idx = fake_data_idx

        self.processed_idx = copy.deepcopy(self.real_data_idx)

    def __len__(self):
        return self.max_num_image // (int(self.batch_size * self.ratio))

    def __iter__(self):
        batch_size_real_data = int(math.floor(self.ratio * self.batch_size))
        batch_size_fake_data = self.batch_size - batch_size_real_data

        self.processed_idx = copy.deepcopy(self.real_data_idx)
        rand_real_data_idx = np.random.permutation(len(self.real_data_idx) // 2)
        for i in range(self.__len__()):
            batch = []
            idx_fake_data = random.sample(self.fake_data_idx, batch_size_fake_data // 2)

            for j in range(batch_size_real_data // 2):
                idx = rand_real_data_idx[(i * batch_size_real_data + j) % (self.real_data_num // 2)]
                batch.append(self.processed_idx[2 * idx])
                batch.append(self.processed_idx[2 * idx + 1])

            for idx in idx_fake_data:
                batch.append(2 * idx + self.real_data_num)
                batch.append(2 * idx + 1 + self.real_data_num)
            yield batch


class SeparateImageList(data.Dataset):
    def __init__(self, real_data_path, real_list_path, fake_data_path, fake_total_num, ratio=0.5):

        self.transform = transforms.Compose([
            transforms.RandomCrop(128),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        # load real nir/vis data
        real_data_list, real_data_idx = self.list_reader(real_data_path, real_list_path)

        # load fake nir/vis data from noise
        _idx = np.random.permutation(fake_total_num)
        fake_data_list = []
        fake_data_idx = []
        for i in range(0, fake_total_num):
            _fake_img_name = str(_idx[i] + 1) + '.jpg'

            # nir_noise and vis_noise are the path of the fake data
            _fake_img_nir_name = 'nir_noise/' + _fake_img_name
            _fake_img_vis_name = 'vis_noise/' + _fake_img_name

            _fake_img_nir_path = os.path.join(fake_data_path, _fake_img_nir_name)
            _fake_img_vis_path = os.path.join(fake_data_path, _fake_img_vis_name)
            fake_data_list.append((_fake_img_nir_path, -1, 0))
            fake_data_list.append((_fake_img_vis_path, -1, 1))
            fake_data_idx.append(i)

        self.real_data_idx = real_data_idx
        self.fake_data_idx = fake_data_idx

        real_data_list.extend(fake_data_list)
        self.all_list = real_data_list

        self.ratio = ratio

        print('real: {}, fake: {}, total: {}, ratio: {}\n'.format(len(self.real_data_idx), len(self.fake_data_idx),
                                                                  len(self.all_list), self.ratio))

    def get_idx(self):
        return self.real_data_idx, self.fake_data_idx

    def list_reader(self, root_path, fileList):
        imgList = []
        imgIdx = []
        img_index = 0
        with open(fileList, 'r') as file:
            for line in file.readlines():
                img_name, label, domain = line.strip().split(' ')
                img_path = os.path.join(root_path, img_name)
                imgList.append((img_path, int(label), int(domain)))
                imgIdx.append(img_index)
                img_index += 1
        return imgList, imgIdx

    def loader(self, path):
        img = Image.open(path).convert('L')
        return img

    def __getitem__(self, index):
        imgPath, target, domain = self.all_list[index]
        img = self.loader(imgPath)

        img = self.transform(img)

        return {'img': img, 'label': target, 'domain_flag': domain}

    def __len__(self):
        return len(self.all_list)

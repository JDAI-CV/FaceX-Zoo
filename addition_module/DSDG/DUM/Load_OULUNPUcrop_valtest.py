from __future__ import print_function, division
import os
import torch
import pandas as pd
# from skimage import io, transform
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os


class Normaliztion_valtest(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label, image_names = sample['image_x'], sample['val_map_x'], sample[
            'spoofing_label'], sample['image_names']
        new_image_x = (image_x - 127.5) / 128  # [-1,1]
        return {'image_x': new_image_x, 'val_map_x': val_map_x, 'spoofing_label': spoofing_label,
                'image_names': image_names}


class ToTensor_valtest(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, val_map_x, spoofing_label, image_names = sample['image_x'], sample['val_map_x'], sample[
            'spoofing_label'], sample['image_names']

        # swap color axis because    BGR2RGB
        # numpy image: (batch_size) x T x H x W x C
        # torch image: (batch_size) x T x C X H X W
        image_x = image_x[:, :, :, ::-1].transpose((0, 3, 1, 2))
        image_x = np.array(image_x)

        val_map_x = np.array(val_map_x)

        spoofing_label_np = np.array([0], dtype=np.long)
        spoofing_label_np[0] = spoofing_label

        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(),
                'val_map_x': torch.from_numpy(val_map_x.astype(np.float)).float(),
                'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long(),
                'image_names': image_names}


class Spoofing_valtest(Dataset):

    def __init__(self, info_list, root_dir, val_map_dir, transform=None):

        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.val_map_dir = val_map_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # print(self.landmarks_frame.iloc[idx, 0])
        videoname = str(self.landmarks_frame.iloc[idx, 1])
        image_path = os.path.join(self.root_dir, videoname)
        val_map_path = os.path.join(self.val_map_dir, videoname)

        image_x, val_map_x, image_names = self.get_single_image_x(image_path, val_map_path, videoname)

        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1  # real
        else:
            spoofing_label = 0

        sample = {'image_x': image_x, 'val_map_x': val_map_x, 'spoofing_label': spoofing_label,
                  'image_names': image_names}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_single_image_x(self, images_path, maps_path, videoname):
        # some vedio flod miss .dat
        files_total = len([name for name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, name))])

        image_x = np.zeros((files_total, 256, 256, 3))
        map_x = np.ones((files_total, 32, 32))
        image_names = []

        file_list = os.listdir(maps_path)

        for i, map_name in enumerate(file_list):
            image_name = map_name[:-12] + '_scene.jpg'
            image_path = os.path.join(images_path, image_name)
            map_path = os.path.join(maps_path, map_name)

            # RGB
            image = cv2.imread(image_path)
            # gray-map
            map = cv2.imread(map_path, 0)

            image_x[i, :, :, :] = cv2.resize(image, (256, 256))
            # transform to binary mask --> threshold = 0 
            map_x[i, :, :] = cv2.resize(map, (32, 32))
            # np.where(temp < 1, temp, 1)
            # val_map_x[i, :, :] = temp

            image_names.append(image_name)

        return image_x, map_x, image_names

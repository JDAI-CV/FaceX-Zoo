"""
@author: Hang Du, Jun Wang
@date: 20201101
@contact: jun21wangustc@gmail.com
"""

import os
import random
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset

def transform(image):
    """ Transform a image by cv2.
    """
    img_size = image.shape[0]
    # random crop
    if random.random() > 0.5:
        crop_size = 9
        x1_offset = np.random.randint(0, crop_size, size=1)[0]
        y1_offset = np.random.randint(0, crop_size, size=1)[0]
        x2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        y2_offset = np.random.randint(img_size-crop_size, img_size, size=1)[0]
        image = image[x1_offset:x2_offset,y1_offset:y2_offset]
        image = cv2.resize(image,(img_size,img_size))
    # horizontal flipping
    if random.random() > 0.5:
        image = cv2.flip(image, 1)
    # grayscale conversion
    if random.random() > 0.8:
        image= cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    # rotation
    if random.random() > 0.5:
        theta = (random.randint(-10,10)) * np.pi / 180
        M_rotate = np.array([[np.cos(theta), -np.sin(theta), 0],[np.sin(theta), np.cos(theta), 0]], dtype=np.float32)
        image = cv2.warpAffine(image, M_rotate, (img_size, img_size))
    # normalizing
    if image.ndim == 2:
        image = (image - 127.5) * 0.0078125
        new_image = np.zeros([3,img_size,img_size], np.float32)
        new_image[0,:,:] = image
        image = torch.from_numpy(new_image.astype(np.float32))
    else:
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
    return image


class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, crop_eye=False):
        self.data_root = data_root
        self.train_list = []
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path, image_label = line.split(' ')
            self.train_list.append((image_path, int(image_label)))
            line = train_file_buf.readline().strip()
        self.crop_eye = crop_eye
    def __len__(self):
        return len(self.train_list)
    def __getitem__(self, index):
        image_path, image_label = self.train_list[index]
        image_path = os.path.join(self.data_root, image_path)
        image = cv2.imread(image_path)
        if self.crop_eye:
            image = image[:60, :]
        #image = cv2.resize(image, (128, 128)) #128 * 128
        if random.random() > 0.5:
            image = cv2.flip(image, 1)
        if image.ndim == 2:
            image = image[:, :, np.newaxis]
        image = (image.transpose((2, 0, 1)) - 127.5) * 0.0078125
        image = torch.from_numpy(image.astype(np.float32))
        return image, image_label

class ImageDataset_SST(Dataset):
    def __init__(self, data_root, train_file, exclude_id_set):
        self.data_root = data_root
        label_set = set()
        # get id2image_path_list
        self.id2image_path_list = {}
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path, label = line.split(' ')
            label = int(label)
            if label in exclude_id_set:
                line = train_file_buf.readline().strip()
                continue
            label_set.add(label)
            if not label in self.id2image_path_list:
                self.id2image_path_list[label] = []
            self.id2image_path_list[label].append(image_path)
            line = train_file_buf.readline().strip()
        self.train_list = list(label_set)
        print('Valid ids: %d.' % len(self.train_list))
            
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        cur_id = self.train_list[index]
        cur_image_path_list = self.id2image_path_list[cur_id]
        if len(cur_image_path_list) == 1:
            image_path1 = cur_image_path_list[0]
            image_path2 = cur_image_path_list[0]
        else:
            training_samples = random.sample(cur_image_path_list, 2)
            image_path1 = training_samples[0]
            image_path2 = training_samples[1]
        image_path1 = os.path.join(self.data_root, image_path1)
        image_path2 = os.path.join(self.data_root, image_path2)
        image1 = cv2.imread(image_path1)
        image2 = cv2.imread(image_path2)
        image1 = transform(image1)
        image2 = transform(image2)
        if random.random() > 0.5:
            return image2, image1, cur_id
        return image1, image2, cur_id


class ImageDataset_HSST(Dataset):
    def __init__(self, data_root, train_file, exclude_id_set):
        self.data_root = data_root
        label_set = set()
        # get id2image_path_nir_list & id2image_path_vis_list
        self.id2image_path_nir_list = {}
        self.id2image_path_vis_list = {}
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            nir_img_path, vis_img_path, label = line.split(' ')
            label = int(label)
            if label in exclude_id_set:
                line = train_file_buf.readline().strip()
                continue
            label_set.add(label)
            if not label in self.id2image_path_nir_list:
                self.id2image_path_nir_list[label] = []
            if not label in self.id2image_path_vis_list:
                self.id2image_path_vis_list[label] = []
            self.id2image_path_nir_list[label].append(nir_img_path)
            self.id2image_path_vis_list[label].append(vis_img_path)
            line = train_file_buf.readline().strip()

        self.train_list = list(label_set)
        
        print('Valid ids: %d.' % len(self.train_list))
            
    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index):
        cur_id = self.train_list[index]
        cur_image_path_nir_list = list(set(self.id2image_path_nir_list[cur_id]))
        cur_image_path_vis_list = list(set(self.id2image_path_vis_list[cur_id]))
        nir_path = random.sample(cur_image_path_nir_list, 1)[0]
        vis_path = random.sample(cur_image_path_vis_list, 1)[0]
        nir_path = os.path.join(self.data_root, nir_path)
        vis_path = os.path.join(self.data_root, vis_path)
        nir_image = cv2.imread(nir_path)
        vis_image = cv2.imread(vis_path)
        nir_image = transform(nir_image)
        vis_image = transform(vis_image)

        return nir_image, vis_image, cur_id

import os
import random

import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from PIL import Image, ImageFile


def read_image(img_path):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    if not os.path.exists(img_path):
        raise IOError("{} does not exist".format(img_path))
    while not got_img:
        try:
            img = Image.open(img_path).convert('RGB')
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(img_path))
            pass
    return img


class ImageDataset(Dataset):
    def __init__(self, data_root, train_file, transform):
        self.data_root = data_root
        self.transform = transform
        self.train_list = []
        train_file_buf = open(train_file)
        line = train_file_buf.readline().strip()
        while line:
            image_path, image_label = line.split(' ')
            self.train_list.append((os.path.join(self.data_root, image_path), int(image_label)))
            line = train_file_buf.readline().strip()

    def __len__(self):
        return len(self.train_list)

    def __getitem__(self, index, transform=None):
        img_path, label = self.train_list[index]
        img = read_image(img_path)

        if self.transform is not None:
            img = self.transform(img)

        return img, label, img_path

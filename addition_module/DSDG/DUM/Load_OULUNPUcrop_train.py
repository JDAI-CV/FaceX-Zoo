from __future__ import print_function, division
import os
import torch
import pandas as pd
import cv2
import numpy as np
import random
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import pdb
import math
import os
import copy
import imgaug.augmenters as iaa

# face_scale = 1.3  #default for test, for training , can be set from [1.2 to 1.5]

# data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
seq = iaa.Sequential([
    iaa.Add(value=(-40, 40), per_channel=True),  # Add color
    iaa.GammaContrast(gamma=(0.5, 1.5))  # GammaContrast with a gamma of 0.5 to 1.5
])


# array
class RandomErasing(object):
    '''
    Class that performs Random Erasing in Random Erasing Data Augmentation by Zhong et al. 
    -------------------------------------------------------------------------------------
    probability: The probability that the operation will be performed.
    sl: min erasing area
    sh: max erasing area
    r1: min aspect ratio
    mean: erasing value
    -------------------------------------------------------------------------------------
    '''

    def __init__(self, probability=0.5, sl=0.01, sh=0.05, r1=0.5, mean=[0.4914, 0.4822, 0.4465]):
        self.probability = probability
        self.mean = mean
        self.sl = sl
        self.sh = sh
        self.r1 = r1

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']

        if random.uniform(0, 1) < self.probability:
            attempts = np.random.randint(1, 3)
            for attempt in range(attempts):
                area = img.shape[0] * img.shape[1]

                target_area = random.uniform(self.sl, self.sh) * area
                aspect_ratio = random.uniform(self.r1, 1 / self.r1)

                h = int(round(math.sqrt(target_area * aspect_ratio)))
                w = int(round(math.sqrt(target_area / aspect_ratio)))

                if w < img.shape[1] and h < img.shape[0]:
                    x1 = random.randint(0, img.shape[0] - h)
                    y1 = random.randint(0, img.shape[1] - w)

                    img[x1:x1 + h, y1:y1 + w, 0] = self.mean[0]
                    img[x1:x1 + h, y1:y1 + w, 1] = self.mean[1]
                    img[x1:x1 + h, y1:y1 + w, 2] = self.mean[2]

        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


# Tensor
class Cutout(object):
    def __init__(self, length=50):
        self.length = length

    def __call__(self, sample):
        img, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']
        h, w = img.shape[1], img.shape[2]  # Tensor [1][2],  nparray [0][1]
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)
        length_new = np.random.randint(1, self.length)

        y1 = np.clip(y - length_new // 2, 0, h)
        y2 = np.clip(y + length_new // 2, 0, h)
        x1 = np.clip(x - length_new // 2, 0, w)
        x2 = np.clip(x + length_new // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return {'image_x': img, 'map_x': map_x, 'spoofing_label': spoofing_label}


class Normaliztion(object):
    """
        same as mxnet, normalize into [-1, 1]
        image = (image - 127.5)/128
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']
        new_image_x = (image_x - 127.5) / 128  # [-1,1]
        new_map_x = map_x / 255.0  # [0,1]
        return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}


class RandomHorizontalFlip(object):
    """Horizontally flip the given Image randomly with a probability of 0.5."""

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']

        new_image_x = np.zeros((256, 256, 3))
        new_map_x = np.zeros((32, 32))

        p = random.random()
        if p < 0.5:
            # print('Flip')

            new_image_x = cv2.flip(image_x, 1)
            new_map_x = cv2.flip(map_x, 1)

            return {'image_x': new_image_x, 'map_x': new_map_x, 'spoofing_label': spoofing_label}
        else:
            # print('no Flip')
            return {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}


class ToTensor(object):
    """
        Convert ndarrays in sample to Tensors.
        process only one batch every time
    """

    def __call__(self, sample):
        image_x, map_x, spoofing_label = sample['image_x'], sample['map_x'], sample['spoofing_label']

        # swap color axis because
        # numpy image: (batch_size) x H x W x C
        # torch image: (batch_size) x C X H X W
        image_x = image_x[:, :, ::-1].transpose((2, 0, 1))
        image_x = np.array(image_x)

        map_x = np.array(map_x)

        spoofing_label_np = np.array([0], dtype=np.long)
        spoofing_label_np[0] = spoofing_label

        return {'image_x': torch.from_numpy(image_x.astype(np.float)).float(),
                'map_x': torch.from_numpy(map_x.astype(np.float)).float(),
                'spoofing_label': torch.from_numpy(spoofing_label_np.astype(np.long)).long()}


class Spoofing_train_g(Dataset):

    def __init__(self, info_list, root_dir, map_dir, transform=None):

        # +1,1_1_21_1
        self.landmarks_frame = pd.read_csv(info_list, delimiter=',', header=None)
        self.root_dir = root_dir
        self.map_dir = map_dir
        self.transform = transform

    def __len__(self):
        return len(self.landmarks_frame)

    def __getitem__(self, idx):
        # 1_1_30_1
        videoname = str(self.landmarks_frame.iloc[idx, 1])
        image_path = os.path.join(self.root_dir, videoname)
        map_path = os.path.join(self.map_dir, videoname)
        image_x, map_x = self.get_single_image_x(image_path, map_path, videoname)

        spoofing_label = self.landmarks_frame.iloc[idx, 0]
        if spoofing_label == 1:
            spoofing_label = 1  # real
        else:
            spoofing_label = 0
            map_x = np.zeros((32, 32))  # fake

        sample = {'image_x': image_x, 'map_x': map_x, 'spoofing_label': spoofing_label}

        if self.transform:
            sample = self.transform(sample)
        return sample

    def get_idx(self):
        real_data_idx = []
        fake_data_idx = []
        i, j = 0, 0
        for idx_all in range(self.__len__()):
            videoname = str(self.landmarks_frame.iloc[idx_all, 1])
            if videoname[:1] == 'p':
                fake_data_idx.append(i)
                i += 1
            else:
                real_data_idx.append(j)
                j += 1
        return real_data_idx, fake_data_idx

    def get_single_image_x(self, images_path, maps_path, videoname):

        frame_total = len([name for name in os.listdir(images_path) if os.path.isfile(os.path.join(images_path, name))])
        # random choose 1 frame

        image_id = np.random.randint(1, frame_total)

        if videoname[:1] == 'p':
            image_id = np.random.randint(1, 100)
            s = "%d_scene" % image_id
            image_name = s + '.jpg'
            # /home/shejiahui5/notespace/data/oulu_img/train_bbox_files/p2_0_1_30/21_scence.jpg
            s = "%d_depth1D" % image_id
            map_name = s + '.jpg'
        else:
            image_id = np.random.randint(1, frame_total)
            s = "_%d_scene" % image_id
            image_name = videoname + s + '.jpg'
            s = "_%d_depth1D" % image_id
            map_name = videoname + s + '.jpg'

        image_path = os.path.join(images_path, image_name)
        map_path = os.path.join(maps_path, map_name)

        map_x = np.zeros((32, 32))

        # RGB
        image_x = cv2.imread(image_path)
        image_x = cv2.resize(image_x, (256, 256))
        # data augment from 'imgaug' --> Add (value=(-40,40), per_channel=True), GammaContrast (gamma=(0.5,1.5))
        image_x_aug = seq.augment_image(image_x)

        # gray-map
        if os.path.exists(map_path):
            map_x = cv2.imread(map_path, 0)
            map_x = cv2.resize(map_x, (32, 32))

        return image_x_aug, map_x


class SeparateBatchSampler(object):
    def __init__(self, real_data_idx, fake_data_idx, batch_size, ratio, put_back=False):
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
            idx_fake_data = random.sample(self.fake_data_idx, batch_size_fake_data)

            for j in range(batch_size_real_data // 2):
                idx = rand_real_data_idx[(i * batch_size_real_data + j) % (self.real_data_num // 2)]
                batch.append(self.processed_idx[2 * idx])
                batch.append(self.processed_idx[2 * idx + 1])

            for idx in idx_fake_data:
                batch.append(idx + self.real_data_num)
                # batch.append(2 * idx + 1 + self.real_data_num)
            yield batch

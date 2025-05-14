# encoding: utf-8
import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader
from .dataset import ImageDataset

def make_transform():
    ori_shape = (256, 256)
    image_crop_size = (224, 224)
    horizontal_flip_p = 0.5
    normalize_std = [0.5, 0.5, 0.5]
    normalize_mean = [0.5, 0.5, 0.5]

    train_transforms = T.Compose([
            T.Resize(ori_shape),
            T.RandomCrop(image_crop_size),
            T.RandomHorizontalFlip(p=horizontal_flip_p),
            T.ToTensor(),
            T.Normalize(mean=normalize_mean, std=normalize_std),
        ])
    val_transforms = T.Compose([
        T.Resize(ori_shape),
        T.CenterCrop(image_crop_size),
        T.ToTensor(),
        T.Normalize(mean=normalize_mean, std=normalize_std)
    ])

    return train_transforms, val_transforms
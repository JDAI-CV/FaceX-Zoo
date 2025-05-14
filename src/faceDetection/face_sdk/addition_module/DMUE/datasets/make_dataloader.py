import torch
import torchvision.transforms as T
from torch.utils.data import DataLoader

from .bases import ImageDataset
from .preprocessing import RandomErasing
from .sampler import ImbalancedDatasetSampler
from .image_utils import add_gaussian_noise, flip_image, color2gray


def make_dataloader(cfg):
    train_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(cfg.ori_shape),
        T.RandomCrop(cfg.image_crop_size),
        T.RandomHorizontalFlip(p=cfg.horizontal_flip_p),
        T.ToTensor(),
        T.Normalize(mean=cfg.normalize_mean,
                    std=cfg.normalize_std),
        T.RandomErasing(scale=(0.02, 0.25))
    ])
    val_transforms = T.Compose([
        T.ToPILImage(),
        T.Resize(cfg.ori_shape),
        T.CenterCrop(cfg.image_crop_size),
        T.ToTensor(),
        T.Normalize(mean=cfg.normalize_mean,
                    std=cfg.normalize_std),
    ])

    train_set = ImageDataset(cfg.train_dataset, transform=train_transforms, lmdb_f=cfg.lmdb_f)
    val_set = ImageDataset(cfg.val_dataset, transform=val_transforms, lmdb_f=cfg.lmdb_f, mode='test')
    # print('Train set size:', train_set.__len__())
    # print('Test set size:', val_set.__len__())

    train_loader = DataLoader(
        train_set,
        sampler=ImbalancedDatasetSampler(train_set),
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers, pin_memory=True
    )

    val_loader = DataLoader(
        val_set, batch_size=cfg.test_minibatch, shuffle=False, num_workers=cfg.num_workers
    )

    return train_loader, val_loader

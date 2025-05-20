import os
import lmdb
import random
import numpy as np

from torch.utils.data import Dataset

from .image_utils import add_gaussian_noise


def read_lmdb(key, txn):
    """Keep reading image until succeed.
    This can avoid IOError incurred by heavy IO process."""
    got_img = False
    while not got_img:
        try:
            img = np.frombuffer(txn.get(key.encode('utf-8')), dtype=np.uint8).reshape(256, 256, 3)  # here fixed
            got_img = True
        except IOError:
            print("IOError incurred when reading '{}'. Will redo. Don't worry. Just chill.".format(key))
            pass
    return img


class ImageDataset(Dataset):
    def __init__(self, dataset, transform=None, lmdb_f=None, mode=None):
        self.dataset = dataset
        self.img_label_list = [x for _, x in self.dataset]
        self.transform = transform
        self.txn = lmdb.open(lmdb_f, readonly=True).begin(write=False)
        self.mode = mode

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        key, label = self.dataset[index]
        img = read_lmdb(key, self.txn)
        
        if self.mode is None:
            if random.uniform(0, 1) > 0.75:
                img = add_gaussian_noise(img)

        if self.transform is not None:
            img = self.transform(img)

        return img, label

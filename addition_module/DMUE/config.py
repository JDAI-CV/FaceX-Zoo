import os
import getpass


def parse_lb_txt(filename):
    lines = open(filename, 'r').readlines()
    train_dataset, test_dataset = [], []
    for line in lines:
        key, label = line.split(' ')[0], line[-2]
        label = int(label)
        mode, img_path = key.split('_') #
        if mode == 'train':
            train_dataset.append([key, label])
        elif mode == 'test':
            test_dataset.append([key, label])

    return train_dataset, test_dataset


class Config:
    num_classes = 8
    ori_shape = (256, 256)
    label_path = "/path/to/your/lb2.txt"
    lmdb_f = "/path/to/your/AffectNet_lmdb/"

    train_dataset, val_dataset = parse_lb_txt(label_path)
    w, T = 0.5, 1.2
    gamma = 1000
    ramp_a = 6 # affectnet 4/6; ferplus 10/12/14; raf 9/10

    batch_size = 72
    test_minibatch=16
    num_workers = 4
    lr1 = [[6, 0.0001], [12, 0.00005], [20, 0.00001], [22, 0.00001], [25, 0.00005], [60, 0.00001]]
    lr2 = [[4, 0.001], [8, 0.0005], [14, 0.0001], [22, 0.00001], [25, 0.00005], [60, 0.00001]]
    bnneck = True # False for resnet50_ibn
    use_dropout = True
    BiasInCls = False
    fc_num = 2
    train_mode = 'sp_confidence' 
    second_order_statics = 'mean' # all, mean, var
    # -----------saving dirs-------#
    ckpt_root_dir = './checkpoints'
    output_dir = 'AffectNet_res18'
    # ---------------------------------------------------------------------------- #
    # Input
    # ---------------------------------------------------------------------------- #
    image_crop_size = (224, 224)
    padding = 0
    image_channel = 3
    horizontal_flip_p = 0.5
    normalize_mean = [0.5, 0.5, 0.5]
    normalize_std = [0.5, 0.5, 0.5]
    # ---------------------------------------------------------------------------- #
    # Model
    # ---------------------------------------------------------------------------- #
    num_branches = num_classes + 1
    assert num_branches == (num_classes + 1)
    backbone = 'resnet18' 
    pretrained = './pretrain/checkpoints/out_dir_res18/mv_epoch_17.pt'
    pretrained_choice = 'msra' # '' or 'msra'
    last_stride = 2
    frozen_stages = -1
    pooling_method = 'GAP'
    # ---------------------------------------------------------------------------- #
    # Optimizer
    # ---------------------------------------------------------------------------- #
    start_epoch = 0
    max_epoch = 36 
    weight_decay = 1e-4

    # set different lr to the backbone and the classifier
    def get_lr(self, epoch):
        for it_lr in self.lr1:
            if epoch < it_lr[0]:
                _lr1 = it_lr[1]
                break
        for it_lr in self.lr2:
            if epoch < it_lr[0]:
                _lr2 = it_lr[1]
                break
        return _lr1, _lr2


config = Config()

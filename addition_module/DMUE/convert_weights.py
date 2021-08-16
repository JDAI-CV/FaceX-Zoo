import os
import argparse
from collections import OrderedDict

import torch


# parser = argparse.ArgumentParser(description='convert multi-branch weights to target branch weights.')
# parser.add_argument("--ori_path", type=str, help="path to multi-branch weights")
# parser.add_argument("--dst_path", type=str, help="path to target branch weights")
# parser.add_argument("--num_classes", type=int, default=8, help="number of expression classes")


def convert(ori_path, dst_path, num_classes):
    num_branches = num_classes + 1
    state_dict = torch.load(ori_path, map_location=lambda storage,loc: storage.cpu())
    new_state_dict = OrderedDict()
    for key in state_dict:
        if 'layer4' in key:
            if 'layer4_{}'.format(num_branches-1) in key:
                new_key = key.replace('module.', '')
                new_key = new_key.replace('layer4_{}'.format(num_branches-1), 'layer4')
                new_state_dict[new_key] = state_dict[key]
            else:
                pass
        elif 'project_w' in key:
            pass
        elif 'classifiers' in key:
            if 'classifiers.{}'.format(num_branches-1) in key:
                new_key = key.replace('module.', '')
                new_key = new_key.replace('{}'.format(num_branches-1), '0')
                new_state_dict[new_key] = state_dict[key]
            else:
                pass
        else:
            new_key = key.replace('module.', '')
            new_state_dict[new_key] = state_dict[key]

    torch.save(obj=new_state_dict, f=dst_path)
    print('Convert Done')


if __name__ == '__main__':
    ori_path = "./checkpoints/AffectNet_res18/snapshots/ep7_b3400_acc0.6285.pth"
    dst_path = "./weights/AffectNet_res18_acc0.6285.pth"
    num_classes = 8

    convert(ori_path, dst_path, num_classes)
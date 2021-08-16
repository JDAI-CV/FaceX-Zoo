import os
import sys
import shutil
import argparse
import logging as logger

import torch
from torch import optim
from torch.utils.data import DataLoader

from loss_def import MVFace
from utils.dataset import ImageDataset
from utils.AverageMeter import AverageMeter
from utils.make_transform import make_transform

parser = argparse.ArgumentParser(description='train mv-softmax on face database.')
# General setting
parser.add_argument("--backbone", type=str, help="resnet18, resnet50 or resnet50_ibn")
parser.add_argument("--data_root", type=str, help="The root folder of train set.")
parser.add_argument("--train_file", type=str,  help="The train file path.")
parser.add_argument("--out_dir", type=str, help="The folder to save models.")
parser.add_argument('--lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--step', type=str, default='10,13,16', help='Step for lr.')
parser.add_argument('--epochs', type=int, default=18, help='The training epoches.')
parser.add_argument('--print_freq', type=int, default=200, help='The print freq for training state.')
parser.add_argument('--batch_size', type=int, default=512, help='The training batch size over all gpus.')
parser.add_argument('--momentum', type=float, default=0.9, help='The momentum for sgd.')
parser.add_argument('--resume', '-r', action='store_true', default=False, help='Resume from checkpoint or not.')
parser.add_argument('--pretrain_model', type=str, default='', help='The path of pretrained model')
# MV-softmax setting
parser.add_argument('--feat_dim', type=int, default=2048, help='Feature dimension, 512/2048 for res18/res50_ibn.')
parser.add_argument('--num_class', type=int, default=86876, help='The number of ids')
parser.add_argument('--is_am', type=int, default=1, help="Use easy margin if 1, otherwise use cosface.")

args = parser.parse_args()
args.milestones = [int(p) for p in args.step.split(',')]
print('Experiments dir: {}'.format(args.out_dir))
if not os.path.exists(args.out_dir):
    os.makedirs(args.out_dir)

log_format = '%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s'
logger.basicConfig(level=logger.INFO, format=log_format, datefmt='%Y-%m-%d %H:%M:%S')
fh = logger.FileHandler(os.path.join(args.out_dir, 'log.txt'))
fh.setFormatter(logger.Formatter(log_format))
logger.getLogger().addHandler(fh)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, model, optimizer, criteria, cur_epoch, loss_meter, args):
    for batch_idx, (images, labels, _) in enumerate(data_loader):
        images = images.cuda()
        labels = labels.cuda()
        labels = labels.squeeze()
        outputs = model.forward(images, labels)
        loss = criteria(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        loss_meter.update(loss.item(), args.batch_size)
        if batch_idx % args.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (cur_epoch, batch_idx, args.db_size, lr, loss_avg))
            global_batch_idx = cur_epoch * args.db_size + batch_idx
            loss_meter.reset()
        if batch_idx != 0 and batch_idx in args.check_point_size:
            saved_name = 'mv_epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': model.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state, os.path.join(args.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)

    saved_name = 'mv_epoch_%d.pt' % cur_epoch
    state = {'state_dict': model.module.state_dict(), 'epoch': cur_epoch, 'batch_id': batch_idx}
    torch.save(state, os.path.join(args.out_dir, saved_name))
    logger.info('Save checkpoint %s to disk...' % saved_name)

def train(args):
    is_am = True if args.is_am == 1 else False
    model = MVFace(args.backbone, args.feat_dim, args.num_class, is_am)

    ori_epoch = 0
    if args.resume:
        ori_epoch = torch.load(args.pretrain_model)['epoch'] + 1
        state_dict = torch.load(args.pretrain_model)['state_dict']
        model.load_state_dict(state_dict)
    model = torch.nn.DataParallel(model).cuda()
    criteria = torch.nn.CrossEntropyLoss().cuda()
    parameters = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(parameters, lr=args.lr, momentum=args.momentum, weight_decay=1e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=0.1)
    train_transform, _ = make_transform()
    data_loader = DataLoader(ImageDataset(args.data_root, args.train_file, train_transform), 
                             args.batch_size, shuffle=True, num_workers=4)
    args.db_size = len(data_loader)
    args.check_point_size = (args.db_size // 3, args.db_size // 2, args.db_size // 3 * 2)
    loss_meter = AverageMeter()
    model.train()
    for epoch in range(ori_epoch, args.epochs):
        train_one_epoch(data_loader, model, optimizer, criteria, epoch, loss_meter, args)
        lr_schedule.step()                        

if __name__ == '__main__':
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')

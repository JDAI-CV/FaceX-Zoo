"""
@author: Jun Wang
@date: 20201019
@contact: jun21wangustc@gmail.com
"""
import os
import sys
import shutil
import argparse
import logging as logger

import torch
import torch.distributed as dist  
import torch.utils.data.distributed
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from apex import amp

from optimizer import build_optimizer
from lr_scheduler import build_scheduler

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')
torch.backends.cudnn.benchmark = True 

class FaceModel(torch.nn.Module):
    """Define a traditional face model which contains a backbone and a head.
    
    Attributes:
        backbone(object): the backbone of face model.
        head(object): the head of face model.
    """
    def __init__(self, backbone_factory, head_factory):
        """Init face model by backbone factorcy and head factory.
        
        Args:
            backbone_factory(object): produce a backbone according to config files.
            head_factory(object): produce a head according to config files.
        """
        super(FaceModel, self).__init__()
        self.backbone = backbone_factory.get_backbone()
        self.head = head_factory.get_head()

    def forward(self, data, label):
        feat = self.backbone.forward(data)
        pred = self.head.forward(feat, label)
        return pred

def get_lr(optimizer):
    """Get the current learning rate from optimizer. 
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def train_one_epoch(data_loader, model, optimizer, lr_schedule, criterion, cur_epoch, loss_meter, args):
    """Tain one epoch by traditional training.
    """
    for batch_idx, (images, labels) in enumerate(data_loader):
        images = images.to(args.local_rank)
        labels = labels.to(args.local_rank)
        labels = labels.squeeze()
        if args.head_type == 'AdaM-Softmax':
            outputs, lamda_lm = model.forward(images, labels)
            lamda_lm = torch.mean(lamda_lm)
            loss = criterion(outputs, labels) + lamda_lm
        else:
            outputs = model.forward(images, labels)
            loss = criterion(outputs, labels)
        optimizer.zero_grad()
        with amp.scale_loss(loss, optimizer) as scaled_loss:
            scaled_loss.backward()
        optimizer.step()
        global_batch_idx = cur_epoch * len(data_loader) + batch_idx
        lr_schedule.step_update(global_batch_idx)
        torch.cuda.synchronize()
        loss_meter.update(loss.item(), images.shape[0])
        if args.local_rank == 0 and batch_idx % args.print_freq == 0:
            loss_avg = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d/%d, lr %f, loss %f' % 
                        (cur_epoch, batch_idx, len(data_loader), lr, loss_avg))
            args.writer.add_scalar('Train_loss', loss_avg, global_batch_idx)
            args.writer.add_scalar('Train_lr', lr, global_batch_idx)
            loss_meter.reset()
        if args.local_rank == 0 and (batch_idx + 1) % args.save_freq == 0:
            saved_name = 'Epoch_%d_batch_%d.pt' % (cur_epoch, batch_idx)
            state = {
                'state_dict': model.module.state_dict(),
                'epoch': cur_epoch,
                'batch_id': batch_idx
            }
            torch.save(state, os.path.join(args.out_dir, saved_name))
            logger.info('Save checkpoint %s to disk.' % saved_name)
        torch.cuda.empty_cache()
    if args.local_rank == 0:
        saved_name = 'Epoch_%d.pt' % cur_epoch
        state = {'state_dict': model.module.state_dict(), 
                 'epoch': cur_epoch, 'batch_id': batch_idx}
        torch.save(state, os.path.join(args.out_dir, saved_name))
        logger.info('Save checkpoint %s to disk...' % saved_name)

def train(args):
    """Total training procedure.
    """
    print("Use GPU: {} for training".format(args.local_rank))
    if args.local_rank == 0:
        writer = SummaryWriter(log_dir=args.tensorboardx_logdir)
        args.writer = writer
        if not os.path.exists(args.out_dir):
            os.makedirs(args.out_dir)
    torch.cuda.set_device(args.local_rank)
    dist.init_process_group(backend='nccl', init_method='env://')
    args.rank = dist.get_rank()
    #print('args.rank: ', dist.get_rank())
    #print('args.get_world_size: ', dist.get_world_size())
    #print('is_nccl_available: ', dist.is_nccl_available())
    args.world_size = dist.get_world_size() 
    trainset = ImageDataset(args.data_root, args.train_file)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True) 
    train_loader = DataLoader(dataset=trainset,
                              batch_size=args.batch_size,
                              sampler=train_sampler,
                              num_workers=0, 
                              pin_memory=True, 
                              drop_last=False) 
    
    backbone_factory = BackboneFactory(args.backbone_type, args.backbone_conf_file)    
    head_factory = HeadFactory(args.head_type, args.head_conf_file)
    model = FaceModel(backbone_factory, head_factory)
    model = model.to(args.local_rank)
    model.train()
    for ps in model.parameters():
        dist.broadcast(ps, 0)
    optimizer = build_optimizer(model, args.lr)
    lr_schedule = build_scheduler(optimizer, len(train_loader), args.epoches, args.warm_up_epoches)
    model, optimizer = amp.initialize(model, optimizer, opt_level="O1")
    # DDP
    model = torch.nn.parallel.DistributedDataParallel(
        module=model,
        broadcast_buffers=False,
        device_ids=[args.local_rank]
    )
    criterion = torch.nn.CrossEntropyLoss().to(args.local_rank)
    loss_meter = AverageMeter()
    model.train()
    ori_epoch = 0
    for epoch in range(ori_epoch, args.epoches):
        train_one_epoch(train_loader, model, optimizer, lr_schedule,
                        criterion, epoch, loss_meter, args)
    dist.destroy_process_group() 

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='traditional_training for face recognition.')
    conf.add_argument('--local_rank', type=int, default=0, help='local_rank')
    conf.add_argument("--data_root", type = str,
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,
                      help = "The training file path.")
    conf.add_argument("--backbone_type", type = str,
                      help = "Mobilefacenets, Resnet.")
    conf.add_argument("--backbone_conf_file", type = str,
                      help = "the path of backbone_conf.yaml.")
    conf.add_argument("--head_type", type = str,
                      help = "mv-softmax, arcface, npc-face.")
    conf.add_argument("--head_conf_file", type = str,
                      help = "the path of head_conf.yaml.")
    conf.add_argument('--lr', type = float, default = 0.1,
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type = str,
                      help = "The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 9,
                      help = 'The training epoches.')
    conf.add_argument('--warm_up_epoches', type = int, default = 9,
                      help = 'The training epoches.')
    '''
    conf.add_argument('--step', type = str, default = '2,5,7',
                      help = 'Step for lr.')
    '''
    conf.add_argument('--print_freq', type = int, default = 10,
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type = int, default = 10,
                      help = 'The save frequency for training state.')
    conf.add_argument('--batch_size', type = int, default = 128,
                      help='The training batch size over all gpus.')
    '''
    conf.add_argument('--momentum', type = float, default = 0.9,
                      help = 'The momentum for sgd.')
    '''
    conf.add_argument('--log_dir', type = str, default = 'log',
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str,
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt',
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False,
                      help = 'Whether to resume from a checkpoint.')    
    args = conf.parse_args()
    #args.milestones = [int(num) for num in args.step.split(',')]
    train(args)

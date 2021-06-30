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
from torch import optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

sys.path.append('../../')
from utils.AverageMeter import AverageMeter
from data_processor.train_dataset import ImageDataset_SST
from backbone.backbone_def import BackboneFactory
from head.head_def import HeadFactory
from test_protocol.utils.online_val import Evaluator

logger.basicConfig(level=logger.INFO, 
                   format='%(levelname)s %(asctime)s %(filename)s: %(lineno)d] %(message)s',
                   datefmt='%Y-%m-%d %H:%M:%S')

def get_lr(optimizer):
    """Get the current learning rate from optimizer.
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def moving_average(probe, gallery, alpha):
    """Update the gallery-set network in the momentum way.(MoCo)
    """
    for param_probe, param_gallery in zip(probe.parameters(), gallery.parameters()):
        param_gallery.data =  \
            alpha* param_gallery.data + (1 - alpha) * param_probe.detach().data

def train_BN(m):
    classname = m.__class__.__name__
    if classname.find('BatchNorm') != -1:
        m.train()
        
def shuffle_BN(batch_size):
    """ShuffleBN for batch, the same as MoCo https://arxiv.org/abs/1911.05722 #######
    """
    shuffle_ids = torch.randperm(batch_size).long().cuda()
    reshuffle_ids = torch.zeros(batch_size).long().cuda()
    reshuffle_ids.index_copy_(0, shuffle_ids, torch.arange(batch_size).long().cuda())
    return shuffle_ids, reshuffle_ids
    
def train_one_epoch(data_loader, probe_net, gallery_net, prototype, optimizer, 
                    criterion, cur_epoch, conf, loss_meter):
    """Tain one epoch by semi-siamese training. 
    """
    for batch_idx, (images1, images2, id_indexes) in enumerate(data_loader):
        batch_size = images1.size(0)
        global_batch_idx = cur_epoch * len(data_loader) + batch_idx
        images1 = images1.cuda()
        images2 = images2.cuda()
        # set inputs as probe or gallery 
        shuffle_ids, reshuffle_ids = shuffle_BN(batch_size)
        images1_probe = probe_net(images1)
        with torch.no_grad():
            images2 = images2[shuffle_ids]
            images2_gallery = gallery_net(images2)[reshuffle_ids]
            images2 = images2[reshuffle_ids]
        shuffle_ids, reshuffle_ids = shuffle_BN(batch_size)
        images2_probe = probe_net(images2)
        with torch.no_grad():
            images1 = images1[shuffle_ids]
            images1_gallery = gallery_net(images1)[reshuffle_ids]
            images1 = images1[reshuffle_ids]
        output1, output2, label, id_set  = prototype(
            images1_probe, images2_gallery, images2_probe, images1_gallery, id_indexes)
        loss = (criterion(output1, label) + criterion(output2, label))/2
        # update
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        moving_average(probe_net, gallery_net, conf.alpha)
        loss_meter.update(loss.item(), batch_size)
        if batch_idx % conf.print_freq == 0:
            loss_val = loss_meter.avg
            lr = get_lr(optimizer)
            logger.info('Epoch %d, iter %d, lr %f, loss %f'  % 
                        (cur_epoch, batch_idx, lr, loss_val))
            conf.writer.add_scalar('Train_loss', loss_val, global_batch_idx)
            conf.writer.add_scalar('Train_lr', lr, global_batch_idx)
    if cur_epoch % conf.save_freq == 0 or cur_epoch == conf.epoches - 1:
        saved_name = ('Epoch_{}.pt'.format(cur_epoch))
        torch.save(probe_net.state_dict(), os.path.join(conf.out_dir, saved_name))
        logger.info('save checkpoint %s to disk...' % saved_name)
    return id_set

def train(conf):
    """Total training procedure. 
    """ 
    conf.device = torch.device('cuda:0')
    criterion = torch.nn.CrossEntropyLoss().cuda(conf.device)
    backbone_factory = BackboneFactory(conf.backbone_type, conf.backbone_conf_file)
    probe_net = backbone_factory.get_backbone()
    gallery_net = backbone_factory.get_backbone()        
    head_factory = HeadFactory(conf.head_type, conf.head_conf_file)
    prototype = head_factory.get_head().cuda(conf.device)
    probe_net = torch.nn.DataParallel(probe_net).cuda()
    gallery_net = torch.nn.DataParallel(gallery_net).cuda()
    optimizer = optim.SGD(probe_net.parameters(), lr=conf.lr, momentum=conf.momentum, weight_decay=5e-4)
    lr_schedule = optim.lr_scheduler.MultiStepLR(optimizer, milestones=conf.milestones, gamma=0.1)
    if conf.resume:
        probe_net.load_state_dict(torch.load(args.pretrain_model))
    moving_average(probe_net, gallery_net, 0)
    probe_net.train()
    gallery_net.eval().apply(train_BN)    

    exclude_id_set = set()
    loss_meter = AverageMeter()
    for epoch in range(conf.epoches):
        data_loader = DataLoader(
            ImageDataset_SST(conf.data_root, conf.train_file, exclude_id_set), 
            conf.batch_size, True, num_workers = 4, drop_last = True)
        exclude_id_set = train_one_epoch(data_loader, probe_net, gallery_net, 
            prototype, optimizer, criterion, epoch, conf, loss_meter)
        lr_schedule.step()
        if conf.evaluate:
            conf.evaluator.evaluate(probe_net)

if __name__ == '__main__':
    conf = argparse.ArgumentParser(description='semi-siamese_training for face recognition.')
    conf.add_argument("--data_root", type = str, 
                      help = "The root folder of training set.")
    conf.add_argument("--train_file", type = str,  
                      help = "The train file path.")
    conf.add_argument('--backbone_type', type=str, default='Mobilefacenets', 
                      help='Mobilefacenets, Resnet.')
    conf.add_argument('--backbone_conf_file', type=str, 
                      help='the path of backbone_conf.yaml.')
    conf.add_argument("--head_type", type = str, 
                      help = "mv-softmax, arcface, npc-face ...")
    conf.add_argument("--head_conf_file", type = str, 
                      help = "the path of head_conf.yaml..")
    conf.add_argument('--lr', type = float, default = 0.1, 
                      help='The initial learning rate.')
    conf.add_argument("--out_dir", type=str, default='out_dir', 
                      help=" The folder to save models.")
    conf.add_argument('--epoches', type = int, default = 130, 
                      help = 'The training epoches.') 
    conf.add_argument('--step', type = str, default = '60,100,120', 
                      help = 'Step for lr.')
    conf.add_argument('--print_freq', type = int, default = 10, 
                      help = 'The print frequency for training state.')
    conf.add_argument('--save_freq', type=int, default=1, 
                      help='The save frequency for training state.')
    conf.add_argument('--batch_size', type=int, default=128, 
                      help='batch size over all gpus.')
    conf.add_argument('--momentum', type=float, default=0.9, 
                      help='The momentum for sgd.')
    conf.add_argument('--alpha', type=float, default=0.999, 
                      help='weight of moving_average')
    conf.add_argument('--log_dir', type = str, default = 'log', 
                      help = 'The directory to save log.log')
    conf.add_argument('--tensorboardx_logdir', type = str, 
                      help = 'The directory to save tensorboardx logs')
    conf.add_argument('--pretrain_model', type = str, default = 'mv_epoch_8.pt', 
                      help = 'The path of pretrained model')
    conf.add_argument('--resume', '-r', action = 'store_true', default = False, 
                      help = 'Resume from checkpoint or not.')
    conf.add_argument('--evaluate', '-e', action = 'store_true', default = False, 
                      help = 'Evaluate the training model.')
    conf.add_argument('--test_set', type = str, default = 'LFW', 
                      help = 'Test set to evaluate the model.')  
    conf.add_argument('--test_data_conf_file', type = str, 
                      help = 'The path of test data conf file.')    
    args = conf.parse_args()
    args.milestones = [int(num) for num in args.step.split(',')]
    if not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir)
    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    tensorboardx_logdir = os.path.join(args.log_dir, args.tensorboardx_logdir)
    if os.path.exists(tensorboardx_logdir):
        shutil.rmtree(tensorboardx_logdir)
    writer = SummaryWriter(log_dir=tensorboardx_logdir)
    args.writer = writer
    if args.evaluate:
        args.evaluator = Evaluator(args.test_set, args.test_data_conf_file)
    logger.info('Start optimization.')
    logger.info(args)
    train(args)
    logger.info('Optimization done!')

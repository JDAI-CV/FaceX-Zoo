from __future__ import print_function, division
import torch
import matplotlib.pyplot as plt
import argparse, os
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms

from models.CDCNs_u import Conv2d_cd, CDCN_u

from Load_OULUNPUcrop_train import Spoofing_train_g, SeparateBatchSampler, Normaliztion, ToTensor, \
    RandomHorizontalFlip, Cutout, RandomErasing
from Load_OULUNPUcrop_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest

import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim

from utils import AvgrageMeter, performances

# Dataset root
train_image_dir = '/export2/home/wht/oulu_img_crop/train_file_flod/'
val_image_dir = '/export2/home/wht/oulu_img_crop/dev_file_flod/'
test_image_dir = '/export2/home/wht/oulu_img_crop/test_file_flod/'

train_map_dir = '/export2/home/wht/oulu_img_crop/train_depth_flod/'
val_map_dir = '/export2/home/wht/oulu_img_crop/dev_depth_flod/'
test_map_dir = '/export2/home/wht/oulu_img_crop/test_depth_flod/'

train_list = '/export2/home/wht/oulu_img_crop/protocols/Protocol_1/Train_g.txt'
val_list = '/export2/home/wht/oulu_img_crop/protocols/Protocol_1/Dev.txt'
test_list = '/export2/home/wht/oulu_img_crop/protocols/Protocol_1/Test.txt'


def contrast_depth_conv(input):
    ''' compute contrast depth in both of (out, label) '''
    '''
        input  32x32
        output 8x32x32
    '''

    kernel_filter_list = [
        [[1, 0, 0], [0, -1, 0], [0, 0, 0]], [[0, 1, 0], [0, -1, 0], [0, 0, 0]], [[0, 0, 1], [0, -1, 0], [0, 0, 0]],
        [[0, 0, 0], [1, -1, 0], [0, 0, 0]], [[0, 0, 0], [0, -1, 1], [0, 0, 0]],
        [[0, 0, 0], [0, -1, 0], [1, 0, 0]], [[0, 0, 0], [0, -1, 0], [0, 1, 0]], [[0, 0, 0], [0, -1, 0], [0, 0, 1]]
    ]

    kernel_filter = np.array(kernel_filter_list, np.float32)

    kernel_filter = torch.from_numpy(kernel_filter.astype(np.float)).float().cuda()
    # weights (in_channel, out_channel, kernel, kernel)
    kernel_filter = kernel_filter.unsqueeze(dim=1)

    input = input.unsqueeze(dim=1).expand(input.shape[0], 8, input.shape[1], input.shape[2])

    contrast_depth = F.conv2d(input, weight=kernel_filter, groups=8)

    return contrast_depth


class Contrast_depth_loss(nn.Module):
    def __init__(self):
        super(Contrast_depth_loss, self).__init__()
        return

    def forward(self, out, label):
        contrast_out = contrast_depth_conv(out)
        contrast_label = contrast_depth_conv(label)

        criterion_MSE = nn.MSELoss().cuda()

        loss = criterion_MSE(contrast_out, contrast_label)

        return loss


def train_test():
    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log + '/' + args.log + '_log_P1.txt', 'a')

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    print('train from scratch!\n')
    log_file.write('train from scratch!\n')
    log_file.write('lr:%.6f, lamda_kl:%.6f , batchsize:%d\n' % (args.lr, args.kl_lambda, args.batchsize))
    log_file.flush()

    model = CDCN_u(basic_conv=Conv2d_cd, theta=0.7)
    # model = ResNet18_u()

    model = model.cuda()
    model = torch.nn.DataParallel(model)

    lr = args.lr
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=0.00005)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    print(model)

    criterion_absolute_loss = nn.MSELoss().cuda()
    criterion_contrastive_loss = Contrast_depth_loss().cuda()

    for epoch in range(args.epochs):
        if (epoch + 1) % args.step_size == 0:
            lr *= args.gamma

        loss_absolute_real = AvgrageMeter()
        loss_absolute_fake = AvgrageMeter()
        loss_contra_real = AvgrageMeter()
        loss_contra_fake = AvgrageMeter()
        loss_kl_real = AvgrageMeter()
        loss_kl_fake = AvgrageMeter()

        ###########################################
        '''                train             '''
        ###########################################
        model.train()

        # load random 16-frame clip data every epoch
        train_data = Spoofing_train_g(train_list, train_image_dir, train_map_dir,
                                      transform=transforms.Compose(
                                          [RandomErasing(), RandomHorizontalFlip(), ToTensor(), Cutout(),
                                           Normaliztion()]))
        train_real_idx, train_fake_idx = train_data.get_idx()
        batch_sampler = SeparateBatchSampler(train_real_idx, train_fake_idx, batch_size=args.batchsize, ratio=args.ratio)
        dataloader_train = DataLoader(train_data, num_workers=8, batch_sampler=batch_sampler)

        for i, sample_batched in enumerate(dataloader_train):
            # get the inputs
            inputs, map_label, spoof_label = sample_batched['image_x'].cuda(), sample_batched['map_x'].cuda(), \
                                             sample_batched['spoofing_label'].cuda()

            optimizer.zero_grad()

            # forward + backward + optimize
            mu, logvar, map_x, x_concat, x_Block1, x_Block2, x_Block3, x_input = model(inputs)

            mu_real = mu[:int(args.batchsize * args.ratio), :, :]
            logvar_real = logvar[:int(args.batchsize * args.ratio), :, :]
            map_x_real = map_x[:int(args.batchsize * args.ratio), :, :]
            map_label_real = map_label[:int(args.batchsize * args.ratio), :, :]

            absolute_loss_real = criterion_absolute_loss(map_x_real, map_label_real)
            contrastive_loss_real = criterion_contrastive_loss(map_x_real, map_label_real)
            kl_loss_real = -(1 + logvar_real - (mu_real - map_label_real).pow(2) - logvar_real.exp()) / 2
            kl_loss_real = kl_loss_real.sum(dim=1).sum(dim=1).mean()
            kl_loss_real = args.kl_lambda * kl_loss_real

            mu_fake = mu[int(args.batchsize * args.ratio):, :, :]
            logvar_fake = logvar[int(args.batchsize * args.ratio):, :, :]
            map_x_fake = map_x[int(args.batchsize * args.ratio):, :, :]
            map_label_fake = map_label[int(args.batchsize * args.ratio):, :, :]

            absolute_loss_fake = 0.1 * criterion_absolute_loss(map_x_fake, map_label_fake)
            contrastive_loss_fake = 0.1 * criterion_contrastive_loss(map_x_fake, map_label_fake)
            kl_loss_fake = -(1 + logvar_fake - (mu_fake - map_label_fake).pow(2) - logvar_fake.exp()) / 2
            kl_loss_fake = kl_loss_fake.sum(dim=1).sum(dim=1).mean()
            kl_loss_fake = 0.1 * args.kl_lambda * kl_loss_fake

            absolute_loss = absolute_loss_real + absolute_loss_fake
            contrastive_loss = contrastive_loss_real + contrastive_loss_fake
            kl_loss = kl_loss_real + kl_loss_fake

            loss = absolute_loss + contrastive_loss + kl_loss

            loss.backward()

            optimizer.step()

            n = inputs.size(0)
            loss_absolute_real.update(absolute_loss_real.data, n)
            loss_absolute_fake.update(absolute_loss_fake.data, n)
            loss_contra_real.update(contrastive_loss_real.data, n)
            loss_contra_fake.update(contrastive_loss_fake.data, n)
            loss_kl_real.update(kl_loss_real.data, n)
            loss_kl_fake.update(kl_loss_fake.data, n)

        scheduler.step()
        # whole epoch average
        print(
            'epoch:%d, Train:  Absolute_loss: real=%.4f,fake=%.4f, '
            'Contrastive_loss: real=%.4f,fake=%.4f, kl_loss: real=%.4f,fake=%.4f' % (
                epoch + 1, loss_absolute_real.avg, loss_absolute_fake.avg, loss_contra_real.avg, loss_contra_fake.avg,
                loss_kl_real.avg, loss_kl_fake.avg))

        # validation/test
        if epoch < 200:
            epoch_test = 200
        else:
            epoch_test = 50
        # epoch_test = 1
        if epoch % epoch_test == epoch_test - 1:
            model.eval()

            with torch.no_grad():
                ###########################################
                '''                val             '''
                ###########################################
                # val for threshold
                val_data = Spoofing_valtest(val_list, val_image_dir, val_map_dir,
                                            transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_val = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=4)

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_val):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    val_maps = sample_batched['val_map_x'].cuda()  # binary map from PRNet

                    optimizer.zero_grad()

                    mu, logvar, map_x, x_concat, x_Block1, x_Block2, x_Block3, x_input = model(inputs.squeeze(0))
                    score_norm = mu.sum(dim=1).sum(dim=1) / val_maps.squeeze(0).sum(dim=1).sum(dim=1)
                    map_score = score_norm.mean()
                    map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))

                map_score_val_filename = args.log + '/' + args.log + '_map_score_val.txt'
                with open(map_score_val_filename, 'w') as file:
                    file.writelines(map_score_list)

                ###########################################
                '''                test             '''
                ##########################################
                # test for ACC
                test_data = Spoofing_valtest(test_list, test_image_dir, test_map_dir,
                                             transform=transforms.Compose([Normaliztion_valtest(), ToTensor_valtest()]))
                dataloader_test = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4)

                map_score_list = []

                for i, sample_batched in enumerate(dataloader_test):
                    # get the inputs
                    inputs, spoof_label = sample_batched['image_x'].cuda(), sample_batched['spoofing_label'].cuda()
                    test_maps = sample_batched['val_map_x'].cuda()

                    optimizer.zero_grad()
                    mu, logvar, map_x, x_concat, x_Block1, x_Block2, x_Block3, x_input = model(inputs.squeeze(0))
                    score_norm = mu.sum(dim=1).sum(dim=1) / test_maps.squeeze(0).sum(dim=1).sum(dim=1)
                    map_score = score_norm.mean()
                    map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))

                map_score_test_filename = args.log + '/' + args.log + '_map_score_test.txt'
                with open(map_score_test_filename, 'w') as file:
                    file.writelines(map_score_list)

                #############################################################
                #       performance measurement both val and test
                #############################################################
                val_threshold, test_threshold, val_ACC, val_ACER, test_ACC, test_APCER, test_BPCER, test_ACER, test_ACER_test_threshold = performances(
                    map_score_val_filename, map_score_test_filename)

                print('epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f' % (
                    epoch + 1, val_threshold, val_ACC, val_ACER))
                log_file.write('\n epoch:%d, Val:  val_threshold= %.4f, val_ACC= %.4f, val_ACER= %.4f \n' % (
                    epoch + 1, val_threshold, val_ACC, val_ACER))

                print('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f' % (
                    epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
                log_file.write('epoch:%d, Test:  ACC= %.4f, APCER= %.4f, BPCER= %.4f, ACER= %.4f \n' % (
                    epoch + 1, test_ACC, test_APCER, test_BPCER, test_ACER))
                log_file.flush()

        if epoch % epoch_test == epoch_test - 1:
            # save the model until the next improvement
            torch.save(model.state_dict(), args.log + '/' + args.log + '_%d.pkl' % (epoch + 1))

    print('Finished Training')
    log_file.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpus', type=str, default='0, 1, 2, 3', help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.0001, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=64, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--kl_lambda', type=float, default=0.001, help='')
    parser.add_argument('--ratio', type=float, default=0.75, help='real and fake in batchsize ')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=1600, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_U_P1", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')

    args = parser.parse_args()
    train_test()

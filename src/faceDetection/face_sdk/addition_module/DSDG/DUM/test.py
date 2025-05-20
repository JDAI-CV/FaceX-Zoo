from __future__ import print_function, division
import torch

torch.set_printoptions(profile="full")
import matplotlib.pyplot as plt
import argparse, os
import numpy as np
import shutil
from torch.utils.data import DataLoader
from torchvision import transforms

from models.CDCNs_u import Conv2d_cd, CDCN_u

from Load_OULUNPUcrop_valtest import Spoofing_valtest, Normaliztion_valtest, ToTensor_valtest

import torch.optim as optim

from utils import performances

# Dataset root

val_image_dir = '/export2/home/wht/oulu_img_crop/dev_file_flod/'
test_image_dir = '/export2/home/wht/oulu_img_crop/test_file_flod/'

val_map_dir = '/export2/home/wht/oulu_img_crop/dev_depth_flod/'
test_map_dir = '/export2/home/wht/oulu_img_crop/test_depth_flod/'

val_list = '/export2/home/wht/oulu_img_crop/protocols/Protocol_1/Dev.txt'
test_list = '/export2/home/wht/oulu_img_crop/protocols/Protocol_1/Test.txt'


# main function
def test():
    # GPU  & log file  -->   if use DataParallel, please comment this command
    os.environ["CUDA_VISIBLE_DEVICES"] = '0, 1, 2, 3'

    isExists = os.path.exists(args.log)
    if not isExists:
        os.makedirs(args.log)
    log_file = open(args.log + '/' + args.log + '_log_P1.txt', 'w')

    log_file.write('Oulu-NPU, P1:\n ')
    log_file.flush()

    print('test!\n')
    log_file.write('test!\n')
    log_file.flush()

    model = CDCN_u(basic_conv=Conv2d_cd, theta=0.7)
    # model = ResNet18_u()

    model = model.cuda()
    model = torch.nn.DataParallel(model)
    model.load_state_dict(torch.load('./DUM/checkpoint/CDCN_U_P1.pkl', map_location='cuda:0'))

    print(model)

    optimizer = optim.Adam(model.parameters(), lr=0.001, weight_decay=0.00005)

    for epoch in range(args.epochs):

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

                # pdb.set_trace()
                map_score = 0.0
                for frame_t in range(inputs.shape[1]):
                    mu, logvar, map_x, x_concat, x_Block1, x_Block2, x_Block3, x_input = model(
                        inputs[:, frame_t, :, :, :])
                    score_norm = torch.sum(mu) / torch.sum(val_maps[:, frame_t, :, :])
                    map_score += score_norm

                map_score = map_score / inputs.shape[1]

                map_score_list.append('{} {}\n'.format(map_score, spoof_label[0][0]))
                # pdb.set_trace()
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

                # pdb.set_trace()
                map_score = 0.0
                for frame_t in range(inputs.shape[1]):
                    mu, logvar, map_x, x_concat, x_Block1, x_Block2, x_Block3, x_input = model(
                        inputs[:, frame_t, :, :, :])

                    score_norm = torch.sum(mu) / torch.sum(test_maps[:, frame_t, :, :])
                    map_score += score_norm

                map_score = map_score / inputs.shape[1]

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

    print('Finished Training')
    log_file.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="save quality using landmarkpose model")
    parser.add_argument('--gpus', type=str, default='0,1,2,3', help='the gpu id used for predict')
    parser.add_argument('--lr', type=float, default=0.001, help='initial learning rate')
    parser.add_argument('--batchsize', type=int, default=32, help='initial batchsize')
    parser.add_argument('--step_size', type=int, default=500, help='how many epochs lr decays once')  # 500
    parser.add_argument('--gamma', type=float, default=0.5, help='gamma of optim.lr_scheduler.StepLR, decay of lr')
    parser.add_argument('--kl_lambda', type=float, default=0.001, help='')
    parser.add_argument('--echo_batches', type=int, default=50, help='how many batches display once')  # 50
    parser.add_argument('--epochs', type=int, default=1, help='total training epochs')
    parser.add_argument('--log', type=str, default="CDCN_U_P1_test", help='log and save model name')
    parser.add_argument('--finetune', action='store_true', default=False, help='whether finetune other models')
    parser.add_argument('--test', action='store_true', default=True, help='')

    args = parser.parse_args()
    test()

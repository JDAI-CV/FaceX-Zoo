import torch
from .generator import Encoder, Decoder, Encoder_s, Decoder_s, Cls
from .light_cnn import network_29layers_v2, resblock


# define generator
def define_G(hdim=256, attack_type=4):
    netE_nir = Encoder_s(hdim=hdim)
    netCls = Cls(hdim=hdim, attack_type=attack_type)
    netE_vis = Encoder(hdim=hdim)
    netG = Decoder_s(hdim=hdim)

    netE_nir = torch.nn.DataParallel(netE_nir).cuda()
    netE_vis = torch.nn.DataParallel(netE_vis).cuda()
    netG = torch.nn.DataParallel(netG).cuda()
    netCls = torch.nn.DataParallel(netCls).cuda()

    return netE_nir, netE_vis, netG, netCls


# define identity preserving && feature extraction net
def define_IP(is_train=False):
    netIP = network_29layers_v2(resblock, [1, 2, 3, 4], is_train)
    netIP = torch.nn.DataParallel(netIP).cuda()
    return netIP


# define recognition network
def LightCNN_29v2(num_classes=10000, is_train=True):
    net = network_29layers_v2(resblock, [1, 2, 3, 4], is_train, num_classes=num_classes)
    net = torch.nn.DataParallel(net).cuda()
    return net

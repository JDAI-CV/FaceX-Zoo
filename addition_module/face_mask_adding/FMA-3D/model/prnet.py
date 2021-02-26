"""
@author: Jun Wang
@date: 20201020
@contact: jun21wangustc@gmail.com
"""

# based on:
# https://github.com/liguohao96/pytorch-prnet

from torch import nn

class Conv2d(nn.Module):
    def __init__(self, input_size, in_channel, out_channel, kernel_size=4, stride=1):
        super(Conv2d, self).__init__()
        output_size = input_size // stride
        self.padding_num = stride * (output_size - 1) - input_size + kernel_size
        self.even_conv = nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                                   stride=stride, padding=self.padding_num // 2, bias=False)
        self.odd_conv = nn.Sequential(
            nn.ConstantPad2d((self.padding_num // 2, self.padding_num // 2 + 1, self.padding_num // 2, self.padding_num // 2 + 1), 0),
            nn.Conv2d(in_channel, out_channel, kernel_size=kernel_size,
                      stride=stride, padding=0, bias=False)
        )
    def forward(self, x):
        if self.padding_num % 2 == 0:
            x = self.even_conv(x)
        else:
            x = self.odd_conv(x)
        return x
            
class UpBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride, activation_fn='relu'):
        super(UpBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.activation_fn = activation_fn
        self.upConvTranspose = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=2, padding=1, bias=False)
        self.convTranspose = nn.Sequential(
            nn.ConstantPad2d((2, 1, 2, 1), 0),
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=4, stride=1, padding=3, bias=False)
        )
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.001)
        self.activation1 = nn.ReLU(inplace=True)
        self.activation2 = nn.Sigmoid()
    def forward(self, x):
        if self.stride == 1:
            x = self.convTranspose(x)
        else:
            x = self.upConvTranspose(x)
        x = self.bn(x)
        if self.activation_fn=='relu':
            x = self.activation1(x)
        else:
            x = self.activation2(x)
        return x
    
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=4, stride=1, input_size=None):
        super(ResBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride        
        self.conv1 = nn.Conv2d(in_channels, out_channels//2, 1, 1, padding=0, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels // 2, eps=0.001, momentum=0.001)

        self.conv2 = Conv2d(input_size, out_channels // 2, out_channels // 2, stride=stride)
        self.bn2 = nn.BatchNorm2d(out_channels // 2, eps=0.001, momentum=0.001)

        input_size = input_size // stride
        self.conv3 = Conv2d(input_size, out_channels // 2, out_channels, kernel_size=1, stride=1)
        
        self.shortcut = nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels, eps=0.001, momentum=0.001)
        self.relu = nn.ReLU(inplace=True) 
    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        if (self.in_channels != self.out_channels) or (self.stride != 1):
            identity = self.shortcut(x)
        out += identity
        out = self.bn3(out)
        out = self.relu(out)
        return out

class PRNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(PRNet, self).__init__()        
        size = 16
        self.input_layer = nn.Sequential(
            Conv2d(256, in_channels, size, kernel_size=4, stride=1),
            nn.BatchNorm2d(size, eps=0.001, momentum=0.001),
            nn.ReLU(inplace=True)
        )
        self.encoder_block_1 = ResBlock(size, size * 2, kernel_size=4, stride=2, input_size=256)  # 128x128x32
        self.encoder_block_2 = ResBlock(size * 2, size * 2, kernel_size=4, stride=1, input_size=128)  # 128x128x32
        self.encoder_block_3 = ResBlock(size * 2, size * 4, kernel_size=4, stride=2, input_size=128)  # 64x64x64
        self.encoder_block_4 = ResBlock(size * 4, size * 4, kernel_size=4, stride=1, input_size=64)  # 64x64x64
        self.encoder_block_5 = ResBlock(size * 4, size * 8, kernel_size=4, stride=2, input_size=64)  # 32x32x128
        self.encoder_block_6 = ResBlock(size * 8, size * 8, kernel_size=4, stride=1, input_size=32)  # 32x32x128
        self.encoder_block_7 = ResBlock(size * 8, size * 16, kernel_size=4, stride=2, input_size=32)  # 16x16x256
        self.encoder_block_8 = ResBlock(size * 16, size * 16, kernel_size=4, stride=1, input_size=16)  # 16x16x256
        self.encoder_block_9 = ResBlock(size * 16, size * 32, kernel_size=4, stride=2, input_size=16)  # 8x8x512
        self.encoder_block_10 = ResBlock(size * 32, size * 32, kernel_size=4, stride=1, input_size=8)  # 8x8x512        

        self.decoder_block_1 = UpBlock(size * 32, size * 32, stride=1) # 8x8x512
        self.decoder_block_2 = UpBlock(size * 32, size * 16, stride=2) # 16x16x256
        self.decoder_block_3 = UpBlock(size * 16, size * 16, stride=1) # 16x16x256
        self.decoder_block_4 = UpBlock(size * 16, size * 16, stride=1) # 16x16x256
        self.decoder_block_5 = UpBlock(size * 16, size * 8, stride=2)# 32 x 32 x 128
        self.decoder_block_6 = UpBlock(size * 8, size * 8, stride=1)# 32 x 32 x 128
        self.decoder_block_7 = UpBlock(size * 8, size * 8, stride=1)# 32 x 32 x 128
        self.decoder_block_8 = UpBlock(size * 8, size * 4, stride=2)# 64 x 64 x 64
        self.decoder_block_9 = UpBlock(size * 4, size * 4, stride=1)# 64 x 64 x 64
        self.decoder_block_10 = UpBlock(size * 4, size * 4, stride=1)# 64 x 64 x 64

        self.decoder_block_11 = UpBlock(size * 4, size * 2, stride=2)# 128 x 128 x 32
        self.decoder_block_12 = UpBlock(size * 2, size * 2, stride=1)# 128 x 128 x 32
        self.decoder_block_13 = UpBlock(size * 2, size, stride=2)# 256 x 256 x 16
        self.decoder_block_14 = UpBlock(size, size, stride=1)# 256 x 256 x 16

        self.decoder_block_15 = UpBlock(size, 3, stride=1)# 256 x 256 x 3
        self.decoder_block_16 = UpBlock(3, 3, stride=1)# 256 x 256 x 3
        self.decoder_block_17 = UpBlock(3, 3, stride=1, activation_fn='sigmoid')#
        
    def forward(self, x):
        x = self.input_layer(x)
        x = self.encoder_block_1(x)
        x = self.encoder_block_2(x)
        x = self.encoder_block_3(x)
        x = self.encoder_block_4(x)
        x = self.encoder_block_5(x)
        x = self.encoder_block_6(x)
        x = self.encoder_block_7(x)
        x = self.encoder_block_8(x)
        x = self.encoder_block_9(x)
        x = self.encoder_block_10(x)
        
        x = self.decoder_block_1(x)
        x = self.decoder_block_2(x)
        x = self.decoder_block_3(x)
        x = self.decoder_block_4(x)
        x = self.decoder_block_5(x)
        x = self.decoder_block_6(x)
        x = self.decoder_block_7(x)
        x = self.decoder_block_8(x)
        x = self.decoder_block_9(x)
        x = self.decoder_block_10(x)
        x = self.decoder_block_11(x)
        x = self.decoder_block_12(x)
        x = self.decoder_block_13(x)
        x = self.decoder_block_14(x)
        x = self.decoder_block_15(x)
        x = self.decoder_block_16(x)
        x = self.decoder_block_17(x)
        return x

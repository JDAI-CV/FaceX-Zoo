"""
@author: Yibo Hu, Jun Wang
@date: 20201019 
@contact: jun21wangustc@gmail.com
"""

import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


def channel_shuffle(x, groups):
	assert groups > 1
	batchsize, num_channels, height, width = x.size()
	assert (num_channels % groups == 0)
	channels_per_group = num_channels // groups
	# reshape
	x = x.view(batchsize, groups, channels_per_group, height, width)
	# transpose
	x = torch.transpose(x, 1, 2).contiguous()
	# flatten
	x = x.view(batchsize, -1, height, width)
	return x


def get_same_padding(kernel_size):
	if isinstance(kernel_size, tuple):
		assert len(kernel_size) == 2, 'invalid kernel size: {}'.format(kernel_size)
		p1 = get_same_padding(kernel_size[0])
		p2 = get_same_padding(kernel_size[1])
		return p1, p2
	assert isinstance(kernel_size, int), 'kernel size should be either `int` or `tuple`'
	assert kernel_size % 2 > 0, 'kernel size should be odd number'
	return kernel_size // 2


class Swish(nn.Module):
	def __init__(self, inplace=False):
		super(Swish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		if self.inplace:
			return x.mul_(x.sigmoid())
		else:
			return x * x.sigmoid()


class HardSwish(nn.Module):
	def __init__(self, inplace=False):
		super(HardSwish, self).__init__()
		self.inplace = inplace

	def forward(self, x):
		if self.inplace:
			return x.mul_(F.relu6(x + 3., inplace=True) / 6.)
		else:
			return x * F.relu6(x + 3.) /6.


class BasicLayer(nn.Module):

	def __init__(
			self,
			in_channels,
			out_channels,
			use_bn=True,
			affine = True,
			act_func='relu6',
			ops_order='weight_bn_act'):
		super(BasicLayer, self).__init__()

		self.in_channels = in_channels
		self.out_channels = out_channels
		self.use_bn = use_bn
		self.affine = affine
		self.act_func = act_func
		self.ops_order = ops_order

		""" add modules """
		# batch norm
		if self.use_bn:
			if self.bn_before_weight:
				self.bn = nn.BatchNorm2d(in_channels, affine=affine, track_running_stats=affine)
			else:
				self.bn = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
		else:
			self.bn = None
		# activation
		if act_func == 'relu':
			if self.ops_list[0] == 'act':
				self.act = nn.ReLU(inplace=False)
			else:
				self.act = nn.ReLU(inplace=True)
		elif act_func == 'relu6':
			if self.ops_list[0] == 'act':
				self.act = nn.ReLU6(inplace=False)
			else:
				self.act = nn.ReLU6(inplace=True)
		elif act_func == 'swish':
			if self.ops_list[0] == 'act':
				self.act = Swish(inplace=False)
			else:
				self.act = Swish(inplace=True)
		elif act_func == 'h-swish':
			if self.ops_list[0] == 'act':
				self.act = HardSwish(inplace=False)
			else:
				self.act = HardSwish(inplace=True)
		else:
			self.act = None

	@property
	def ops_list(self):
		return self.ops_order.split('_')

	@property
	def bn_before_weight(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'weight':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)

	def weight_call(self, x):
		raise NotImplementedError

	def forward(self, x):
		for op in self.ops_list:
			if op == 'weight':
				x = self.weight_call(x)
			elif op == 'bn':
				if self.bn is not None:
					x = self.bn(x)
			elif op == 'act':
				if self.act is not None:
					x = self.act(x)
			else:
				raise ValueError('Unrecognized op: %s' % op)
		return x


class ConvLayer(BasicLayer):

	def __init__(
			self,
			in_channels,
			out_channels,
			kernel_size=3,
			stride=1,
			groups=1,
			has_shuffle=False,
			bias=False,
			use_bn=True,
			affine=True,
			act_func='relu6',
			ops_order='weight_bn_act'):
		super(ConvLayer, self).__init__(
			in_channels,
			out_channels,
			use_bn,
			affine,
			act_func,
			ops_order)

		self.kernel_size = kernel_size
		self.stride = stride
		self.groups = groups
		self.has_shuffle = has_shuffle
		self.bias = bias

		padding = get_same_padding(self.kernel_size)
		self.conv = nn.Conv2d(
			in_channels,
			out_channels,
			kernel_size=self.kernel_size,
			stride=self.stride,
			padding=padding,
			groups=self.groups,
			bias=self.bias)

	def weight_call(self, x):
		x = self.conv(x)
		if self.has_shuffle and self.groups > 1:
			x = channel_shuffle(x, self.groups)
		return x


class LinearLayer(nn.Module):

	def __init__(
			self,
			in_features,
			out_features,
			bias=True,
			use_bn=False,
			affine=False,
			act_func=None,
			ops_order='weight_bn_act'):
		super(LinearLayer, self).__init__()

		self.in_features = in_features
		self.out_features = out_features
		self.bias = bias
		self.use_bn = use_bn
		self.affine = affine
		self.act_func = act_func
		self.ops_order = ops_order

		""" add modules """
		# batch norm
		if self.use_bn:
			if self.bn_before_weight:
				self.bn = nn.BatchNorm1d(in_features, affine=affine, track_running_stats=affine)
			else:
				self.bn = nn.BatchNorm1d(out_features, affine=affine, track_running_stats=affine)
		else:
			self.bn = None
		# activation
		if act_func == 'relu':
			if self.ops_list[0] == 'act':
				self.act = nn.ReLU(inplace=False)
			else:
				self.act = nn.ReLU(inplace=True)
		elif act_func == 'relu6':
			if self.ops_list[0] == 'act':
				self.act = nn.ReLU6(inplace=False)
			else:
				self.act = nn.ReLU6(inplace=True)
		elif act_func == 'tanh':
			self.act = nn.Tanh()
		elif act_func == 'sigmoid':
			self.act = nn.Sigmoid()
		else:
			self.act = None
		# linear
		self.linear = nn.Linear(self.in_features, self.out_features, self.bias)

	@property
	def ops_list(self):
		return self.ops_order.split('_')

	@property
	def bn_before_weight(self):
		for op in self.ops_list:
			if op == 'bn':
				return True
			elif op == 'weight':
				return False
		raise ValueError('Invalid ops_order: %s' % self.ops_order)

	def forward(self, x):
		for op in self.ops_list:
			if op == 'weight':
				x = self.linear(x)
			elif op == 'bn':
				if self.bn is not None:
					x = self.bn(x)
			elif op == 'act':
				if self.act is not None:
					x = self.act(x)
			else:
				raise ValueError('Unrecognized op: %s' % op)
		return x


class MBInvertedResBlock(nn.Module):

	def __init__(
			self,
			in_channels,
			mid_channels,
			se_channels,
			out_channels,
			kernel_size=3,
			stride=1,
			groups=1,
			has_shuffle=False,
			bias=False,
			use_bn=True,
			affine=True,
			act_func='relu6'):
		super(MBInvertedResBlock, self).__init__()

		self.in_channels = in_channels
		self.mid_channels = mid_channels
		self.se_channels = se_channels
		self.out_channels = out_channels
		self.kernel_size = kernel_size
		self.stride = stride
		self.groups = groups
		self.has_shuffle = has_shuffle
		self.bias = bias
		self.use_bn = use_bn
		self.affine = affine
		self.act_func = act_func

		# inverted bottleneck
		if mid_channels > in_channels:
			inverted_bottleneck = OrderedDict([
					('conv', nn.Conv2d(in_channels, mid_channels, 1, 1, 0, groups=groups, bias=bias)),
				])
			if use_bn:
				inverted_bottleneck['bn'] = nn.BatchNorm2d(mid_channels, affine=affine, track_running_stats=affine)
			if act_func == 'relu':
				inverted_bottleneck['act'] = nn.ReLU(inplace=True)
			elif act_func == 'relu6':
				inverted_bottleneck['act'] = nn.ReLU6(inplace=True)
			elif act_func == 'swish':
				inverted_bottleneck['act'] = Swish(inplace=True)
			elif act_func == 'h-swish':
				inverted_bottleneck['act'] = HardSwish(inplace=True)
			self.inverted_bottleneck = nn.Sequential(inverted_bottleneck)
		else:
			self.inverted_bottleneck = None
			self.mid_channels = in_channels
			mid_channels = in_channels

		# depthwise convolution
		padding = get_same_padding(self.kernel_size)
		depth_conv = OrderedDict([
				('conv', 
				 nn.Conv2d(
				 	 mid_channels,
				 	 mid_channels,
				 	 kernel_size,
				 	 stride,
				 	 padding,
				 	 groups=mid_channels,
				 	 bias=bias)),
			])
		if use_bn:
			depth_conv['bn'] = nn.BatchNorm2d(mid_channels, affine=affine, track_running_stats=affine)
		if act_func == 'relu':
			depth_conv['act'] = nn.ReLU(inplace=True)
		elif act_func == 'relu6':
			depth_conv['act'] = nn.ReLU6(inplace=True)
		elif act_func == 'swish':
			depth_conv['act'] = Swish(inplace=True)
		elif act_func == 'h-swish':
			depth_conv['act'] = HardSwish(inplace=True)
		self.depth_conv = nn.Sequential(depth_conv)

		# se model
		if se_channels > 0:
			squeeze_excite = OrderedDict([
					('conv_reduce', nn.Conv2d(mid_channels, se_channels, 1, 1, 0, groups=groups, bias=True)),
				])
			if act_func == 'relu':
				squeeze_excite['act'] = nn.ReLU(inplace=True)
			elif act_func == 'relu6':
				squeeze_excite['act'] = nn.ReLU6(inplace=True)
			elif act_func == 'swish':
				squeeze_excite['act'] = Swish(inplace=True)
			elif act_func == 'h-swish':
				squeeze_excite['act'] = HardSwish(inplace=True)
			squeeze_excite['conv_expand'] = nn.Conv2d(se_channels, mid_channels, 1, 1, 0, groups=groups, bias=True)
			self.squeeze_excite = nn.Sequential(squeeze_excite)
		else:
			self.squeeze_excite = None
			self.se_channels = 0

		# pointwise linear
		point_linear = OrderedDict([
				('conv', nn.Conv2d(mid_channels, out_channels, 1, 1, 0, groups=groups, bias=bias)),
			])
		if use_bn:
			point_linear['bn'] = nn.BatchNorm2d(out_channels, affine=affine, track_running_stats=affine)
		self.point_linear = nn.Sequential(point_linear)

		# residual flag
		self.has_residual = (in_channels == out_channels) and (stride == 1)

	def forward(self, x):
		res = x

		if self.inverted_bottleneck is not None:
			x = self.inverted_bottleneck(x)
			if self.has_shuffle and self.groups > 1:
				x = channel_shuffle(x, self.groups)

		x = self.depth_conv(x)
		if self.squeeze_excite is not None:
			x_se = F.adaptive_avg_pool2d(x, 1)
			x = x * torch.sigmoid(self.squeeze_excite(x_se))

		x = self.point_linear(x)
		if self.has_shuffle and self.groups > 1:
			x = channel_shuffle(x, self.groups)

		if self.has_residual:
			x += res

		return x

class Flatten(nn.Module):
	def forward(self, x):
		return x.view(x.size(0), -1)


class TF_NAS_A(nn.Module):
	def __init__(self, out_h, out_w, feat_dim, drop_ratio=0.0):
		super(TF_NAS_A, self).__init__()
		self.drop_ratio = drop_ratio

		self.first_stem  = ConvLayer(3, 32, kernel_size=3, stride=1, act_func='relu')
		self.second_stem = MBInvertedResBlock(32, 32, 8, 16, kernel_size=3, stride=1, act_func='relu')
		self.stage1 = nn.Sequential(
				MBInvertedResBlock(16, 83, 32, 24, kernel_size=3, stride=2, act_func='relu'),
				MBInvertedResBlock(24, 128, 0, 24, kernel_size=5, stride=1, act_func='relu'),
			)
		self.stage2 = nn.Sequential(
				MBInvertedResBlock(24, 138, 48, 40, kernel_size=3, stride=2, act_func='swish'),
				MBInvertedResBlock(40, 297, 0,  40, kernel_size=3, stride=1, act_func='swish'),
				MBInvertedResBlock(40, 170, 80, 40, kernel_size=5, stride=1, act_func='swish'),
			)
		self.stage3 = nn.Sequential(
				MBInvertedResBlock(40, 248, 80, 80, kernel_size=5, stride=2, act_func='swish'),
				MBInvertedResBlock(80, 500, 0,  80, kernel_size=3, stride=1, act_func='swish'),
				MBInvertedResBlock(80, 424, 0,  80, kernel_size=3, stride=1, act_func='swish'),
				MBInvertedResBlock(80, 477, 0,  80, kernel_size=3, stride=1, act_func='swish'),
			)
		self.stage4 = nn.Sequential(
				MBInvertedResBlock(80,  504, 160, 112, kernel_size=3, stride=1, act_func='swish'),
				MBInvertedResBlock(112, 796, 0,   112, kernel_size=3, stride=1, act_func='swish'),
				MBInvertedResBlock(112, 723, 224, 112, kernel_size=3, stride=1, act_func='swish'),
				MBInvertedResBlock(112, 555, 224, 112, kernel_size=3, stride=1, act_func='swish'),
			)
		self.stage5 = nn.Sequential(
				MBInvertedResBlock(112, 813,  0,   192, kernel_size=3, stride=2, act_func='swish'),
				MBInvertedResBlock(192, 1370, 0,   192, kernel_size=3, stride=1, act_func='swish'),
				MBInvertedResBlock(192, 1138, 384, 192, kernel_size=3, stride=1, act_func='swish'),
				MBInvertedResBlock(192, 1359, 384, 192, kernel_size=3, stride=1, act_func='swish'),
			)
		self.stage6 = nn.Sequential(
				MBInvertedResBlock(192, 1203, 384, 320, kernel_size=5, stride=1, act_func='swish'),
			)
		self.feature_mix_layer = ConvLayer(320, 1280, kernel_size=1, stride=1, act_func='none')
		self.output_layer = nn.Sequential(
			nn.Dropout(self.drop_ratio),
			Flatten(),
			nn.Linear(1280 * out_h * out_w, feat_dim),
			nn.BatchNorm1d(feat_dim))

		self._initialization()

	def forward(self, x):
                x = self.first_stem(x)
                x = self.second_stem(x)
                for block in self.stage1:
                        x = block(x)
                for block in self.stage2:
                        x = block(x)
                for block in self.stage3:
                        x = block(x)
                for block in self.stage4:
                        x = block(x)
                for block in self.stage5:
                        x = block(x)
                for block in self.stage6:
                        x = block(x)
                x = self.feature_mix_layer(x)
                x = self.output_layer(x)
                return x

	def _initialization(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.Linear):
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				if m.weight is not None:
					nn.init.constant_(m.weight, 1)
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)


if __name__ == '__main__':
	x = torch.rand((2,3,112,112))
	net = TF_NAS_A(7, 7, 512, drop_ratio=0.0)

	x = x.cuda()
	net = net.cuda()

	out = net(x)
	print(out.size())

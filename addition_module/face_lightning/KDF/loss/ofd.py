from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


'''
Modified from https://github.com/clovaai/overhaul-distillation/blob/master/CIFAR-100/distiller.py
'''
class OFD(nn.Module):
	'''
	A Comprehensive Overhaul of Feature Distillation
	http://openaccess.thecvf.com/content_ICCV_2019/papers/
	Heo_A_Comprehensive_Overhaul_of_Feature_Distillation_ICCV_2019_paper.pdf
	'''
	def __init__(self, in_channels, out_channels):
		super(OFD, self).__init__()
		self.connector = nn.Sequential(*[
				nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0, bias=False),
				nn.BatchNorm2d(out_channels)
			])

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			elif isinstance(m, nn.BatchNorm2d):
				nn.init.constant_(m.weight, 1)
				nn.init.constant_(m.bias, 0)

	def forward(self, fm_s, fm_t):
		margin = self.get_margin(fm_t)
		fm_t = torch.max(fm_t, margin)
		fm_s = self.connector(fm_s)

		mask = 1.0 - ((fm_s <= fm_t) & (fm_t <= 0.0)).float()
		loss = torch.mean((fm_s - fm_t)**2 * mask)

		return loss

	def get_margin(self, fm, eps=1e-6):
		mask = (fm < 0.0).float()
		masked_fm = fm * mask

		margin = masked_fm.sum(dim=(0,2,3), keepdim=True) / (mask.sum(dim=(0,2,3), keepdim=True)+eps)

		return margin
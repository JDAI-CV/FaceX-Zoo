from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


def conv1x1(in_channels, out_channels):
	return nn.Conv2d(in_channels, out_channels,
					 kernel_size=1, stride=1,
					 padding=0, bias=False)

'''
Modified from https://github.com/HobbitLong/RepDistiller/blob/master/distiller_zoo/VID.py
'''
class VID(nn.Module):
	'''
	Variational Information Distillation for Knowledge Transfer
	https://zpascal.net/cvpr2019/Ahn_Variational_Information_Distillation_for_Knowledge_Transfer_CVPR_2019_paper.pdf
	'''
	def __init__(self, in_channels, mid_channels, out_channels, init_var, eps=1e-6):
		super(VID, self).__init__()
		self.eps = eps
		self.regressor = nn.Sequential(*[
				conv1x1(in_channels, mid_channels),
				# nn.BatchNorm2d(mid_channels),
				nn.ReLU(),
				conv1x1(mid_channels, mid_channels),
				# nn.BatchNorm2d(mid_channels),
				nn.ReLU(),
				conv1x1(mid_channels, out_channels),
			])
		self.alpha = nn.Parameter(
				np.log(np.exp(init_var-eps)-1.0) * torch.ones(out_channels)
			)

		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
				if m.bias is not None:
					nn.init.constant_(m.bias, 0)
			# elif isinstance(m, nn.BatchNorm2d):
			# 	nn.init.constant_(m.weight, 1)
			# 	nn.init.constant_(m.bias, 0)

	def forward(self, fm_s, fm_t):
		pred_mean = self.regressor(fm_s)
		pred_var  = torch.log(1.0+torch.exp(self.alpha)) + self.eps
		pred_var  = pred_var.view(1, -1, 1, 1)
		neg_log_prob = 0.5 * (torch.log(pred_var) + (pred_mean-fm_t)**2 / pred_var)
		loss = torch.mean(neg_log_prob)

		return loss
